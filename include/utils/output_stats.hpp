#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <ctime>
#include <iostream>
#include <map>
#include <fstream>
#include <utils/xml_util.hpp>
#include <utils/cmdline.hpp>
#include <utils/stopwatch.hpp>
#include <data_types/header.hpp>
#include "cuda.h"
#include <iomanip> // for std::setprecision
#include <sstream> // for std::stringstream


class OutputFileWriter {
  XML::Element root;

public:
  OutputFileWriter()
    :root("peasoup_search"){}

  std::string to_string(void){
    return root.to_string(true);
  }

  std::string format_float_to_precision(float value, int precision) {
    std::stringstream stream;
    stream << std::fixed << std::setprecision(precision) << value;
    return stream.str();
}

  void to_file(std::string filename){
    std::ofstream outfile;
    outfile.open(filename.c_str(),std::ifstream::out | std::ifstream::binary);
    ErrorChecker::check_file_error(outfile, filename);
    outfile << root.to_string(true);
    ErrorChecker::check_file_error(outfile, filename);
    outfile.close();
  }
  
  void add_header(std::string filename){
    std::ifstream infile;
    SigprocHeader hdr;
    infile.open(filename.c_str(),std::ifstream::in | std::ifstream::binary);
    ErrorChecker::check_file_error(infile, filename);
    read_header(infile,hdr);
    XML::Element header("header_parameters");
    header.append(XML::Element("source_name",hdr.source_name));
    header.append(XML::Element("rawdatafile",hdr.rawdatafile));
    header.append(XML::Element("az_start",hdr.az_start));
    header.append(XML::Element("za_start",hdr.za_start));
    header.append(XML::Element("src_raj",hdr.src_raj));
    header.append(XML::Element("src_dej",hdr.src_dej));
    header.append(XML::Element("tstart",hdr.tstart));
    header.append(XML::Element("tsamp",hdr.tsamp));
    header.append(XML::Element("period",hdr.period));
    header.append(XML::Element("fch1",hdr.fch1));
    header.append(XML::Element("foff",hdr.foff));
    header.append(XML::Element("nchans",hdr.nchans));
    header.append(XML::Element("telescope_id",hdr.telescope_id));
    header.append(XML::Element("machine_id",hdr.machine_id));
    header.append(XML::Element("data_type",hdr.data_type));
    header.append(XML::Element("ibeam",hdr.ibeam));
    header.append(XML::Element("nbeams",hdr.nbeams));
    header.append(XML::Element("nbits",hdr.nbits));
    header.append(XML::Element("barycentric",hdr.barycentric));
    header.append(XML::Element("pulsarcentric",hdr.pulsarcentric));
    header.append(XML::Element("nbins",hdr.nbins));
    header.append(XML::Element("nsamples",hdr.nsamples));
    header.append(XML::Element("nifs",hdr.nifs));
    header.append(XML::Element("npuls",hdr.npuls));
    header.append(XML::Element("refdm",hdr.refdm));
    header.append(XML::Element("signed",(int)hdr.signed_data));
    root.append(header);
  }

  void add_segment_parameters(SigprocFilterbank& f, CmdLineOptions& args){
    XML::Element segment_parameters("segment_parameters");
    segment_parameters.append(XML::Element("segment_start_sample", f.get_start_sample()));
    segment_parameters.append(XML::Element("segment_nsamples", f.get_effective_nsamps()));

    if (args.keplerian_tb_file != "none") {
            segment_parameters.append(XML::Element("segment_pepoch", f.get_segment_pepoch_template_bank()));
    } else {
            segment_parameters.append(XML::Element("segment_pepoch", f.get_segment_pepoch_accel_search()));
    }
    root.append(segment_parameters);
  }

  void add_search_parameters(CmdLineOptions& args, const Keplerian_TemplateBank_Reader* tb_reader = nullptr){
    XML::Element search_options("search_parameters");
    search_options.append(XML::Element("infilename",args.infilename));
    search_options.append(XML::Element("outdir",args.outdir));
    search_options.append(XML::Element("killfilename",args.killfilename));
    search_options.append(XML::Element("zapfilename",args.zapfilename));
    search_options.append(XML::Element("max_num_threads",args.max_num_threads));
    search_options.append(XML::Element("size",args.size));
    search_options.append(XML::Element("dmfilename",args.dm_file));
    search_options.append(XML::Element("cdm",format_float_to_precision(args.cdm, 4)));
    search_options.append(XML::Element("dm_start",args.dm_start));
    search_options.append(XML::Element("dm_end",args.dm_end));
    search_options.append(XML::Element("dm_tol",args.dm_tol));
    search_options.append(XML::Element("dm_pulse_width",args.dm_pulse_width));
    search_options.append(XML::Element("acc_start",args.acc_start));
    search_options.append(XML::Element("acc_end",args.acc_end));
    search_options.append(XML::Element("acc_tol",args.acc_tol));
    search_options.append(XML::Element("acc_pulse_width",args.acc_pulse_width));
    search_options.append(XML::Element("boundary_5_freq",args.boundary_5_freq));
    search_options.append(XML::Element("boundary_25_freq",args.boundary_25_freq));
    search_options.append(XML::Element("nharmonics",args.nharmonics));
    search_options.append(XML::Element("npdmp",args.npdmp));
    search_options.append(XML::Element("min_snr",args.min_snr));
    search_options.append(XML::Element("min_freq",args.min_freq));
    search_options.append(XML::Element("max_freq",args.max_freq));
    search_options.append(XML::Element("max_harm",args.max_harm));
    search_options.append(XML::Element("freq_tol",args.freq_tol));
    search_options.append(XML::Element("verbose",args.verbose));
    search_options.append(XML::Element("progress_bar",args.progress_bar));
    search_options.append(XML::Element("template_bank_file", args.keplerian_tb_file));
    
    if (tb_reader != nullptr) {
      for (auto const& kv : tb_reader->get_metadata()) {
        std::string raw_key = kv.first;     
        std::string raw_val = kv.second;    
        // Clean the key:
        std::string cleaned_key;
        for (char c : raw_key) {
          if (std::isspace((unsigned char)c)) {
            cleaned_key.push_back('_');
          }
          else if (c == '(' || c == ')') {
            // drop parentheses
          }
          else {
            cleaned_key.push_back(
              static_cast<char>(std::tolower((unsigned char)c)));
          }
        }
        search_options.append(XML::Element(cleaned_key, raw_val));
      }
    }
    
    root.append(search_options);
  }

  void add_misc_info(void){
    XML::Element info("misc_info");
    char buf[128];
    getlogin_r(buf,128);
    std::time_t t = std::time(NULL);
    std::strftime(buf, 128, "%Y-%m-%d-%H:%M", std::localtime(&t));
    info.append(XML::Element("local_datetime",buf));
    std::strftime(buf, 128, "%Y-%m-%d-%H:%M", std::gmtime(&t));
    info.append(XML::Element("utc_datetime",buf));
    root.append(info);
  }
  
  void add_timing_info(std::map<std::string,Stopwatch>& elapsed_times){
    XML::Element times("execution_times");
    typedef std::map<std::string,Stopwatch>::iterator it_type;
    for (it_type it=elapsed_times.begin(); it!=elapsed_times.end(); it++)
      times.append(XML::Element(it->first,it->second.getTime()));
    root.append(times);
  }
  
  void add_gpu_info(std::vector<int>& device_idxs){
    XML::Element gpu_info("cuda_device_parameters");
    int runtime_version,driver_version;
    cudaRuntimeGetVersion(&runtime_version);
    cudaDriverGetVersion(&driver_version);
    gpu_info.append(XML::Element("runtime",runtime_version));
    gpu_info.append(XML::Element("driver",driver_version));
    cudaDeviceProp properties;
    for (int ii=0;ii<device_idxs.size();ii++){
      XML::Element device("cuda_device");
      device.add_attribute("id",device_idxs[ii]);
      cudaGetDeviceProperties(&properties,device_idxs[ii]);
      device.append(XML::Element("name",properties.name));
      device.append(XML::Element("major_cc",properties.major));
      device.append(XML::Element("minor_cc",properties.minor));
      gpu_info.append(device);
    }
    root.append(gpu_info);
  }
  
  void add_dm_list(std::vector<float>& dms){
    XML::Element dm_trials("dedispersion_trials");
    dm_trials.add_attribute("count", dms.size());
    for (int ii = 0; ii < dms.size(); ii++) {
        XML::Element trial("trial");
        trial.add_attribute("id", ii);
        
        // Format the DM value to 4 decimal places
        std::stringstream formatted_value;
        formatted_value << std::fixed << std::setprecision(4) << dms[ii];
        
        // Set the formatted string as the text for the XML element
        trial.set_text(formatted_value.str());
        dm_trials.append(trial);
    }
    root.append(dm_trials);
}
  
  void add_acc_list(std::vector<float>& accs, float cdm){
    XML::Element acc_trials("acceleration_trials");
    acc_trials.add_attribute("count",accs.size());
    acc_trials.add_attribute("DM", format_float_to_precision(cdm, 4));
    for(int ii=0;ii<accs.size();ii++){
      XML::Element trial("trial");
      trial.add_attribute("id",ii);
      trial.set_text(accs[ii]);
      acc_trials.append(trial);
    }
    root.append(acc_trials);
  }

  void add_candidates(std::vector<Candidate>& candidates, 
                    const std::map<unsigned, long int>& byte_map, 
                    SigprocFilterbank& f)
{
    XML::Element cands("candidates");
    constexpr double RAD2DEG = 180.0 / M_PI;
    constexpr double TWO_PI = 2 * M_PI;

    for (std::size_t ii = 0; ii < candidates.size(); ++ii) {
        const Candidate& cand_ref = candidates[ii];

        XML::Element cand("candidate");

        double pb_days = (cand_ref.n > 0) ? TWO_PI / (cand_ref.n * 86400.0) : 0.0;
        double phi_normalised = (cand_ref.phi > 0.0) ? (cand_ref.phi / TWO_PI) : 0.0;
        double T0 = (pb_days > 0.0) ? f.get_segment_pepoch_template_bank() + phi_normalised * pb_days : 0.0;
        double phi_deg = (cand_ref.phi > 0.0) ? (cand_ref.phi * RAD2DEG) : 0.0;
        double omega_deg = (cand_ref.omega > 0.0) ? (cand_ref.omega * RAD2DEG) : 0.0;

        cand.add_attribute("id", static_cast<int>(ii));
        cand.append(XML::Element("period", 1.0 / cand_ref.freq));
        cand.append(XML::Element("opt_period", cand_ref.opt_period));
        cand.append(XML::Element("dm", cand_ref.dm));
        cand.append(XML::Element("acc", cand_ref.acc));
        cand.append(XML::Element("jerk", cand_ref.jerk));
        cand.append(XML::Element("pb", pb_days));
        cand.append(XML::Element("a1", cand_ref.a1));
        cand.append(XML::Element("phi", phi_deg));
        cand.append(XML::Element("t0", T0));
        cand.append(XML::Element("omega", omega_deg));
        cand.append(XML::Element("ecc", cand_ref.ecc));
        cand.append(XML::Element("nh", cand_ref.nh));
        cand.append(XML::Element("snr", cand_ref.snr));
        cand.append(XML::Element("folded_snr", cand_ref.folded_snr));
        cand.append(XML::Element("is_adjacent", cand_ref.is_adjacent));
        cand.append(XML::Element("is_physical", cand_ref.is_physical));
        cand.append(XML::Element("ddm_count_ratio", cand_ref.ddm_count_ratio));
        cand.append(XML::Element("ddm_snr_ratio", cand_ref.ddm_snr_ratio));
        cand.append(XML::Element("nassoc", candidates[ii].count_assoc()));
        cands.append(cand);
    }

    root.append(cands);
}


  void add_candidates(std::vector<Candidate>& candidates,
		      std::map<int,std::string>& filenames){
    XML::Element cands("candidates");
    for (int ii=0;ii<candidates.size();ii++){
      XML::Element cand("candidate");
      cand.add_attribute("id",ii);
      cand.append(XML::Element("period",1.0/candidates[ii].freq));
      cand.append(XML::Element("opt_period",candidates[ii].opt_period));
      cand.append(XML::Element("dm",candidates[ii].dm));
      cand.append(XML::Element("acc",candidates[ii].acc));
      cand.append(XML::Element("nh",candidates[ii].nh));
      cand.append(XML::Element("snr",candidates[ii].snr));
      cand.append(XML::Element("folded_snr",candidates[ii].folded_snr));
      cand.append(XML::Element("is_adjacent",candidates[ii].is_adjacent));
      cand.append(XML::Element("is_physical",candidates[ii].is_physical));
      cand.append(XML::Element("ddm_count_ratio",candidates[ii].ddm_count_ratio));
      cand.append(XML::Element("ddm_snr_ratio",candidates[ii].ddm_snr_ratio));
      cand.append(XML::Element("nassoc",candidates[ii].count_assoc()));
      cand.append(XML::Element("results_file",filenames[ii]));
      cands.append(cand);
    }    
    root.append(cands);
  }
 
};


class CandidateFileWriter {
public:
  std::map<int,std::string> filenames;
  std::map<unsigned,long int> byte_mapping;
  std::string output_dir;
 
  CandidateFileWriter(std::string output_directory)
    :output_dir(output_directory)
  {
    struct stat st = {0};
    if (stat(output_dir.c_str(), &st) == -1) {
      if (mkdir(output_dir.c_str(), 0777) != 0)
	perror(output_dir.c_str());	
    }
  }

  void write_binary(std::vector<Candidate>& candidates,
		    std::string filename)
  {
    char actualpath [PATH_MAX];
    std::stringstream filepath;
    filepath << output_dir << "/" << filename;
        
    FILE* fo = fopen(actualpath,"w");
    if (fo == NULL) {
      perror(filepath.str().c_str());
      return;
    }
    
    for (int ii=0;ii<candidates.size();ii++)
      {
	byte_mapping[ii] = ftell(fo);
	if (candidates[ii].fold.size()>0)
	  {
	    size_t size = candidates[ii].nbins * candidates[ii].nints;
	    float* fold = &candidates[ii].fold[0];
	    fprintf(fo,"FOLD");
	    fwrite(&candidates[ii].nbins,sizeof(int),1,fo);
	    fwrite(&candidates[ii].nints,sizeof(int),1,fo);
	    fwrite(fold,sizeof(float),size,fo);
	  }
	std::vector<CandidatePOD> detections;
	candidates[ii].collect_candidates(detections);
	int ndets = detections.size();
	fwrite(&ndets,sizeof(int),1,fo);
	fwrite(&detections[0],sizeof(CandidatePOD),ndets,fo);
      }
    fclose(fo);

    if (realpath(filepath.str().c_str(), actualpath) == NULL) {
        perror("Binary candidate file realpath error");
        return;
    }
  }
  
  void write_binaries(std::vector<Candidate>& candidates)
  {
    char actualpath [PATH_MAX];
    char filename[1024];
    std::stringstream filepath;
    for (int ii=0;ii<candidates.size();ii++){
      filepath.str("");
      sprintf(filename,"cand_%04d_%.5f_%.1f_%.1f.peasoup",
              ii,1.0/candidates[ii].freq,candidates[ii].dm,candidates[ii].acc);
      filepath << output_dir << "/" << filename;

      char* ptr = realpath(filepath.str().c_str(), actualpath);
      filenames[ii] = std::string(actualpath);
      
      FILE* fo = fopen(filepath.str().c_str(),"w");
      if (fo == NULL) {
	perror(filepath.str().c_str());
	return;
      }
      
      if (candidates[ii].fold.size()>0){
	size_t size = candidates[ii].nbins * candidates[ii].nints;
	float* fold = &candidates[ii].fold[0];
	fprintf(fo,"FOLD");
	fwrite(&candidates[ii].nbins,sizeof(int),1,fo);
	fwrite(&candidates[ii].nints,sizeof(int),1,fo);
	fwrite(fold,sizeof(float),size,fo);
      }
      std::vector<CandidatePOD> detections;
      candidates[ii].collect_candidates(detections);
      int ndets = detections.size();
      fwrite(&ndets,sizeof(int),1,fo);
      fwrite(&detections[0],sizeof(CandidatePOD),ndets,fo);
      fclose(fo);
    }
  }
};


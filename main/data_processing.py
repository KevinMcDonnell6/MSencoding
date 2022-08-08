

import numpy as np
import re    
import calculate_fragments as cf
import model_utils

def within_tol(x, y, atol, rtol):
    x = np.asanyarray(x)
    y = np.asanyarray(y)
    diff = x-y
    logic = np.less_equal(abs(diff), atol + rtol * np.abs(y))
    log_dif = np.zeros((*logic.shape,2))
    log_dif[...,0] = logic
    log_dif[...,1] = diff
    return log_dif 
    
    

def sigmoid(x):
    
    return 1/(1+np.exp(-x))

def get_random_stack(data,stack_size):

        indices = np.random.randint(low=0,
                                    high=len(data),
                                    size=stack_size)

        return([data[i] for i in indices])


def adjacency_matrix(peaks,pep_mass,relative_tolerance=1e-5,absolute_tolerance=1e-8,ion_type=None):
    
    
    peak_masses = peaks[:,0]
    peak_ints = peaks[:,1]
    peak_ints_np = np.array(peak_ints)
    
    peak_diffs = np.subtract(peak_masses[...,np.newaxis],peak_masses)
    peak_sums = np.add(peak_masses[...,np.newaxis],peak_masses)
    
    
    compare_array = np.concatenate((peak_diffs,
                                    pep_mass-peak_sums,
                                    peak_sums-pep_mass
                                    ),0)
    
    AAs = model_utils.mass_ID[3:]
    AA_list = []
    
    for i,aa in enumerate(AAs):
        r,c = np.where(np.isclose(-aa,
                                    compare_array,
                                    rtol=relative_tolerance,
                                    atol=absolute_tolerance))
        num = len(r)
        peak_int_sum = zip(peak_ints_np[r%len(peak_masses)], peak_ints_np[c])
        AA_list += [arr for arr in zip(r%len(peak_masses),c,[i]*num,peak_int_sum)]

    AA_list = sorted(AA_list, key= lambda t: (t[0],t[1],t[2]))
     
    return AA_list


def AA_search_8ion(peaks,prefix_masses,pep_mass,direction="fw"):
    
    if direction=="fw":
        sign=1
    elif direction=="bw":
        sign=-1
    else:
        raise ValueError("Invalid direction. direction should be 'fw' or 'bw'")
        
    AAs = model_utils.mass_ID
    peak_masses = peaks[...,0]
    peak_ints = peaks[...,1]
    
    window_size = model_utils.WINDOW_SIZE
    bin_size = 0.01
    
    peak_diffs = np.subtract(peak_masses[...,np.newaxis],prefix_masses)
    peak_sums = np.add(peak_masses[...,np.newaxis],prefix_masses)
    
    peak_diffs_H2O = peak_diffs+model_utils.mass_H2O
    peak_diffs_NH3 = peak_diffs+model_utils.mass_NH3
    peak_diffs_plus2 = np.subtract(2*peak_masses[...,np.newaxis]-model_utils.mass_H,prefix_masses)
    
    peak_sums_H2O = peak_sums+model_utils.mass_H2O
    peak_sums_NH3 = peak_sums+model_utils.mass_NH3
    peak_sums_plus2 = np.add(2*peak_masses[...,np.newaxis]-model_utils.mass_H,prefix_masses)
    
    compare_array = np.concatenate((peak_diffs,
                                    peak_diffs_H2O,
                                    peak_diffs_NH3,
                                    peak_diffs_plus2,
                                    pep_mass-peak_sums,
                                    pep_mass-peak_sums_H2O,
                                    pep_mass-peak_sums_NH3,
                                    pep_mass-peak_sums_plus2,
                                    ),0)
    

    cand_int = np.zeros((len(prefix_masses),
                         model_utils.vocab_size,
                         len(compare_array)//len(peak_masses),# number of ion types
                         window_size))
    
    for i,aa in enumerate(AAs):
        
        log_diff = within_tol(sign*aa,
                              compare_array,
                              atol=(bin_size*window_size/2)-.0001,
                              rtol = 0)
        
        
        c,r = np.where(log_diff[...,0])

        AA_diffs = np.negative(log_diff[c,r][...,1])

        cand_int[r,i,c//len(peak_masses),((window_size//2)+np.floor(AA_diffs*(window_size))).astype(np.int32)] = peak_ints[c%len(peak_masses)]
        
    cand_int[:,:3,:,:].fill(0.0) # PAD, GO and EOS to 0
    return cand_int



def get_rank(x,normalise=None, pepmass=None):

    N = x.size
    
    x_arg_sort = np.argsort(-x)
    x_rank = x_arg_sort.argsort()
  
    half_rank = N-np.searchsorted(np.flip(x[x_arg_sort]),x/2)
    
    return x_rank, half_rank
    

def local_peaks(x,window_size=50):
    x_mz, x_int = np.transpose(x)
    left_side = x_mz-window_size
    right_side = x_mz+window_size
    
    local_rank = []
    local_half_rank = []
    for i,j,intensity in zip(left_side,right_side,x_int):
        l=np.searchsorted(x_mz,i)
        r=np.searchsorted(x_mz,j,side="right")
        bin_size = r-l
        _bin = x[l:r]
        _bin = _bin[_bin[...,1].argsort()]
        local_rank.append(bin_size-np.searchsorted(_bin[...,1], intensity))
        local_half_rank.append(bin_size-np.searchsorted(_bin[...,1], intensity/2))
        
    
    return local_rank,local_half_rank



    
def get_ions_all(peak,peaks,peptide_mass):
   
    b_H2O = peaks[...,0]+model_utils.mass_H2O
    b_NH3 = peaks[...,0]+model_utils.mass_NH3
    b_plus2 = (2*peaks[...,0])-model_utils.mass_H
    
    y = peptide_mass-peaks[...,0]
    y_H2O = peptide_mass-peaks[...,0]-model_utils.mass_H2O
    y_NH3 = peptide_mass-peaks[...,0]-model_utils.mass_NH3
    y_plus2 = peptide_mass-(2*peaks[...,0]-model_utils.mass_H)
    
    compare_array = np.concatenate((y,
                                    b_H2O,
                                    y_H2O,
                                    b_NH3,
                                    y_NH3,
                                    b_plus2,
                                    y_plus2))
    
    ion_matches = np.ones([model_utils.num_ion-1,2])*[1,0]
    
    log_diff = within_tol(peak[0],compare_array, atol=model_utils.AA_mass_tolerance-1e-6, rtol=0.0)
        
    match_idxs = np.where(log_diff[...,0])[0]
    order = np.argsort(log_diff[match_idxs,1])
    diff_int = np.stack((log_diff[:,1][match_idxs[order]],peaks[...,1][match_idxs[order]%len(peaks)]),-1)
    ion_matches[match_idxs[order]//len(peaks)] = diff_int
    return ion_matches
    
    
    
def convert_ion2window(opp_ion):
    num_ions = int(opp_ion.shape[-1]/2)
    window_size = model_utils.WINDOW_SIZE
    
    windows = np.zeros((len(opp_ion),num_ions,window_size))
    tol=model_utils.AA_mass_tolerance
    
    indices = np.where(opp_ion[...,1::2]) 
    diffs = opp_ion[...,::2][indices]
    ints = opp_ion[...,1::2][indices]
    
    window_indices = (np.floor((tol+diffs)*(window_size)/(2*tol))).astype(int)
    windows[(*indices,window_indices)] = ints
    
    return windows


def ion_peak_present(peaks,peptide_mass):
   
    ion_list = []
    
    for peak in peaks:
        all_ions = []
        all_ions = get_ions_all(peak, peaks, peptide_mass)
        ion_list.append(all_ions)

    assert len(peaks) == len(ion_list)
    
    return np.array(ion_list)


def match_peaks(peptide_ids,spectrum_mz, tolerance,
                _8ions=False):
    
    peptide_str = [model_utils.vocab_reverse[i] for i in peptide_ids]
    if _8ions:
        true_peaks,tp_ions,tp_frags = cf.fragments_mgf_8ion(peptide=peptide_str)
    else:
        true_peaks,tp_ions,tp_frags = cf.fragments_mgf(peptide=peptide_str)
        
    matched_peaks = []
    matched_diffs = []
    for mz in spectrum_mz:
        log_diff = within_tol(mz,true_peaks,rtol=0,atol=tolerance)
        
        if np.any(log_diff[...,0]):
            closest_idx = np.argmin(np.abs(log_diff[...,1]))
            matched_peaks.append(1)
            matched_diffs.append(log_diff[...,1][closest_idx])
                                  
        else:
            matched_peaks.append(0)
            matched_diffs.append(0)    
    return matched_peaks,matched_diffs




def V2diffs(peaks,pep_mass,direction="fw"):
    
    num_ions = model_utils.num_ion_tnet
    
    AAs = model_utils.mass_ID
    peak_masses = peaks[...,0]
    
    if direction=="fw":
        b_locs = np.add(peak_masses[...,np.newaxis],AAs)
        y_locs = np.subtract(pep_mass,b_locs)
        
    elif direction=="bw":
        b_locs = np.subtract(peak_masses[...,np.newaxis],AAs)
        y_locs = np.subtract(pep_mass,b_locs)
    else:
        raise ValueError("Invalid direction. direction should be 'fw' or 'bw'")
    
    cand_loc = np.zeros((num_ions,len(peaks),len(AAs)))
    cand_loc[0] = b_locs
    cand_loc[4] = y_locs
    
    
    b_H2O_locs = b_locs - model_utils.mass_H2O
    b_NH3_locs = b_locs - model_utils.mass_NH3
    b_plus2_locs = (b_locs+model_utils.mass_H)/2
    cand_loc[1] = b_H2O_locs
    cand_loc[2] = b_NH3_locs
    cand_loc[3] = b_plus2_locs
    
    
    y_H2O_locs = y_locs - model_utils.mass_H2O
    y_NH3_locs = y_locs - model_utils.mass_NH3
    y_plus2_locs = (y_locs+model_utils.mass_H)/2
    cand_loc[5] = y_H2O_locs
    cand_loc[6] = y_NH3_locs
    cand_loc[7] = y_plus2_locs
    
    if num_ions == 12:
        a_locs = b_locs - model_utils.mass_CO
        cand_loc[8] = a_locs
        a_H2O_locs = a_locs - model_utils.mass_H2O
        a_NH3_locs = a_locs - model_utils.mass_NH3
        a_plus2_locs = (a_locs+model_utils.mass_H)/2
        cand_loc[9] = a_H2O_locs
        cand_loc[10] = a_NH3_locs
        cand_loc[11] = a_plus2_locs
    
    mask = np.logical_and(
        cand_loc>0,
        cand_loc<=model_utils.MZ_MAX).astype(np.float32)
    cand_loc = cand_loc * mask
    return np.transpose(cand_loc,(1,2,0))



def V2diffs_self(peaks,pep_mass):
    
    num_ions = 7
    
    peak_masses = peaks[...,0]
    
    cand_loc = np.zeros((num_ions,len(peaks)))
    cand_loc[0] = pep_mass-peak_masses 
    cand_loc[1] = peak_masses - model_utils.mass_H2O 
    cand_loc[2] = pep_mass-peak_masses - model_utils.mass_H2O 
    cand_loc[3] = peak_masses - model_utils.mass_NH3
    cand_loc[4] = pep_mass-peak_masses - model_utils.mass_NH3
    cand_loc[5] = (peak_masses+model_utils.mass_H)/2 
    cand_loc[6] = (pep_mass-peak_masses+model_utils.mass_H)/2 
    
    return np.transpose(cand_loc)
    
    
 
def process_spectrum_Graph(spectrum):
    
    scan,                  \
    peptide_ids,            \
    spectrum_mz,             \
    spectrum_intensity,       \
    peptide_mass,              \
    pep_charge                  = spectrum
    
    max_num_peaks = model_utils.max_num_peaks

    AA_edge_precision = model_utils.AA_mass_tolerance
    
    Mp  = peptide_mass
    
    spectrum_intensity = np.divide(spectrum_intensity,max(spectrum_intensity))
    
    peaks = np.stack([spectrum_mz,spectrum_intensity],axis=1)
    
    
    b_or_y, diffs = match_peaks(peptide_ids=peptide_ids, 
                         spectrum_mz=spectrum_mz, 
                         tolerance=model_utils.AA_mass_tolerance)
    
    
    mask = np.concatenate((np.ones(len(b_or_y)),np.zeros(max_num_peaks)))
    
   
    b_or_y = np.pad(b_or_y,(0,max_num_peaks))
    
           
    peaks_forward = np.concatenate((peaks,
                                    [[model_utils.mass_N_terminus,1.0]],
                                    [[Mp+model_utils.mass_N_terminus,1.0]],   
                                    [[model_utils.mass_C_terminus+(2*model_utils.mass_H),1.0]],
                                    [[Mp+model_utils.mass_C_terminus+(2*model_utils.mass_H),1.0]]
                                    ))
    
    cand_int = AA_search_8ion(peaks=peaks_forward, 
                              prefix_masses= peaks[...,0], 
                              pep_mass=peptide_mass+model_utils.mass_N_terminus+model_utils.mass_C_terminus+(2*model_utils.mass_H))
    
    cand_int_bw = AA_search_8ion(peaks=peaks_forward, 
                              prefix_masses= peaks[...,0], 
                              pep_mass=peptide_mass+model_utils.mass_N_terminus+model_utils.mass_C_terminus+(2*model_utils.mass_H),
                              direction="bw")
    
    
    opp_ion = ion_peak_present(peaks,peptide_mass+
                                      model_utils.mass_N_terminus+
                                      model_utils.mass_C_terminus+
                                      (2*model_utils.mass_H))
    
    opp_ion = np.reshape(opp_ion,[opp_ion.shape[0],-1])
    
    ion_windows = convert_ion2window(opp_ion)
    
    peak_rank, peak_half_rank = get_rank(peaks[...,1])
    
    local_rank, local_half_rank = local_peaks(peaks)
    
    
    features = np.stack([peak_rank,peak_half_rank,local_rank,local_half_rank],axis=1)
    
    
    AA_list = adjacency_matrix(peaks_forward[:max_num_peaks],
                                              pep_mass = peptide_mass+model_utils.mass_N_terminus+model_utils.mass_C_terminus+(2*model_utils.mass_H),
                                    absolute_tolerance=AA_edge_precision
                                    )
    
    if AA_list == []:
        d0,d1,d2,vals = [[],[],[],[0,0]]
    else:
        d0,d1,d2,vals = list(zip(*AA_list))
    
    
    cand_int_pad_fw = np.zeros(([max_num_peaks,model_utils.vocab_size,model_utils.num_ion,model_utils.WINDOW_SIZE]))
    cand_int_pad_bw = np.zeros(([max_num_peaks,model_utils.vocab_size,model_utils.num_ion,model_utils.WINDOW_SIZE]))
    features_pad = np.zeros((max_num_peaks,*features.shape[1:]))
    ion_windows_pad = np.zeros((max_num_peaks,*ion_windows.shape[1:]))
    
        
    cand_int_pad_fw[:len(cand_int[:max_num_peaks])] = cand_int[:max_num_peaks]
    cand_int_pad_bw[:len(cand_int_bw[:max_num_peaks])] = cand_int_bw[:max_num_peaks]
    features_pad[:len(features[:max_num_peaks])] = features[:max_num_peaks]
    ion_windows_pad[:len(ion_windows[:max_num_peaks])] = ion_windows[:max_num_peaks]
    
    return [b_or_y[:max_num_peaks],
            mask[:max_num_peaks],
            cand_int_pad_fw,
            cand_int_pad_bw,
            features_pad,
            ion_windows_pad,
            [(d0,d1,d2),vals],
            scan
            ]



def process_spectrum_Tnet(spectrum):
    
    scan,                  \
    peptide_ids,            \
    spectrum_mz,             \
    spectrum_intensity,       \
    peptide_mass,              \
    pep_charge                  = spectrum
    

    max_num_peaks = model_utils.max_num_peaks

    AA_edge_precision = model_utils.AA_mass_tolerance
    
    Mp  = peptide_mass

    spectrum_intensity = np.divide(spectrum_intensity,max(spectrum_intensity))

    peaks = np.stack([spectrum_mz,spectrum_intensity],axis=1)
    
    
    b_or_y, diffs = match_peaks(peptide_ids=peptide_ids, 
                         spectrum_mz=spectrum_mz, 
                         tolerance=model_utils.AA_mass_tolerance)
    
    mask = np.concatenate((np.ones(len(b_or_y)),np.zeros(max_num_peaks)))
    
    b_or_y = np.pad(b_or_y,(0,max_num_peaks))
    
            
    peaks_forward = np.concatenate((peaks,
                                    [[model_utils.mass_N_terminus,1.0]],
                                    [[Mp+model_utils.mass_N_terminus,1.0]],
                                    [[model_utils.mass_C_terminus+(2*model_utils.mass_H),1.0]],
                                    [[Mp+model_utils.mass_C_terminus+(2*model_utils.mass_H),1.0]]
                                    ))
    
    ion_locs = V2diffs(peaks_forward, 
                           peptide_mass+
                            model_utils.mass_N_terminus+
                            model_utils.mass_C_terminus+
                            (2*model_utils.mass_H))
    ion_locs_bw = V2diffs(peaks_forward, 
                           peptide_mass+
                            model_utils.mass_N_terminus+
                            model_utils.mass_C_terminus+
                            (2*model_utils.mass_H),
                            direction="bw") 
    
    ion_locs_self = V2diffs_self(peaks_forward, 
                           peptide_mass+
                            model_utils.mass_N_terminus+
                            model_utils.mass_C_terminus+
                            (2*model_utils.mass_H))
    
    peak_rank, peak_half_rank = get_rank(peaks[...,1])
    
    local_rank, local_half_rank = local_peaks(peaks)
    
    features = np.stack([peak_rank,peak_half_rank,local_rank,local_half_rank],axis=1)
    
    
    ion_locs_pad_fw = np.zeros(([max_num_peaks,model_utils.vocab_size,model_utils.num_ion_tnet]))
    ion_locs_pad_bw = np.zeros(([max_num_peaks,model_utils.vocab_size,model_utils.num_ion_tnet]))
    ion_locs_pad_self = np.zeros(([max_num_peaks,7]))
    
    features_pad = np.zeros((max_num_peaks,*features.shape[1:]))
        
    ion_locs_pad_fw[:len(ion_locs[:max_num_peaks])] = ion_locs[:max_num_peaks]
    ion_locs_pad_bw[:len(ion_locs_bw[:max_num_peaks])] = ion_locs_bw[:max_num_peaks]
    ion_locs_pad_self[:len(ion_locs_self[:max_num_peaks])] = ion_locs_self[:max_num_peaks]
    
    features_pad[:len(features[:max_num_peaks])] = features[:max_num_peaks]
    peaks = np.pad(peaks,((0,max_num_peaks),(0,0)))
    
    return [b_or_y[:max_num_peaks],
            mask[:max_num_peaks],
            ion_locs_pad_fw,
            ion_locs_pad_bw,
            peaks[:max_num_peaks],
            features_pad,
            ion_locs_pad_self,
            scan
            ]


def inspect_file_location(data_format, input_file):
  if (data_format=="msp"):
    keyword="Name" 
  elif (data_format=="mgf"):
    keyword="BEGIN IONS" 

  spectra_file_location = []
  with open(input_file, mode="r") as file_handle:
    line = True
    while line:
      file_location = file_handle.tell()
      line = file_handle.readline()
      if keyword in line:
        spectra_file_location.append(file_location)
  return spectra_file_location





def read_spectra_only(file_handle, data_format, spectra_locations, artificial=False):
    
  data_set = []

  max_seq_len = model_utils.max_seq_len-2
  
  counter = 0
  counter_skipped = 0
  counter_skipped_mod = 0
  counter_skipped_len = 0
  counter_skipped_mass = 0
  counter_skipped_mass_precision = 0
  
  avg_peak_count = 0.0
  avg_peptide_len = 0.0

  if (data_format=="mgf"):
    keyword="BEGIN IONS" 

  for location in spectra_locations:
    
    file_handle.seek(location)
    line = file_handle.readline()
    assert (keyword in line), "ERROR: read_spectra(); wrong format"
  
    unknown_modification = False

    # READ AN ENTRY
    if (data_format=="mgf"):

      # header TITLE
      line = file_handle.readline()

      # header PEPMASS
      line = file_handle.readline()
      peptide_ion_mz = float(re.split('=|\n', line)[1])

      # header CHARGE
      line = file_handle.readline()
      charge = float(re.split('=|\+', line)[1])
      
      # header SCANS
      line = file_handle.readline()
#~       scan = int(re.split('=', line)[1])
      scan = re.split('=|\n', line)[1]

      # header RTINSECONDS
      line = file_handle.readline()

      # header SEQ
      line = file_handle.readline()
      raw_sequence = re.split('=|\n|\r', line)[1]
      raw_sequence_len = len(raw_sequence)
      peptide = []
      index = 0
      while (index < raw_sequence_len):
        if (raw_sequence[index]=="("):
          if (peptide[-1]=="C" and raw_sequence[index:index+8]=="(+57.02)"):
            peptide[-1] = "Cmod"
            index += 8
          elif (peptide[-1]=='M' and raw_sequence[index:index+8]=="(+15.99)"):
            peptide[-1] = 'Mmod'
            index += 8
          elif (peptide[-1]=='N' and raw_sequence[index:index+6]=="(+.98)"):
            peptide[-1] = 'Nmod'
            index += 6
          elif (peptide[-1]=='Q' and raw_sequence[index:index+6]=="(+.98)"):
            peptide[-1] = 'Qmod'
            index += 6
          else: 
            unknown_modification = True
            break
        else:
          peptide.append(raw_sequence[index])
          index += 1
      #
      if (unknown_modification):
        counter_skipped += 1
        counter_skipped_mod += 1
        continue
        
      peptide_mass = peptide_ion_mz*charge - charge*model_utils.mass_H -model_utils.mass_N_terminus-model_utils.mass_C_terminus
      
    
      if (peptide_mass > model_utils.MZ_MAX):
        counter_skipped += 1
        counter_skipped_mass += 1
        continue

      peptide_len = len(peptide)
      if (peptide_len > max_seq_len): 
        counter_skipped += 1
        counter_skipped_len += 1
        continue

      sequence_mass = sum(model_utils.mass_AA[aa] for aa in peptide)
      if (abs(peptide_mass-sequence_mass) > model_utils.PRECURSOR_MASS_PRECISION_INPUT_FILTER):
        counter_skipped_mass_precision += 1
        counter_skipped += 1
        continue

      spectrum_mz = []
      spectrum_intensity = []
      line = file_handle.readline()
      while not ("END IONS" in line):
        
        mz, intensity = re.split(' |\n|\t', line)[:2]
        intensity_float = float(intensity)
        mz_float = float(mz)
        
        if (mz_float > model_utils.MZ_MAX): 
          line = file_handle.readline()
          continue
        
        spectrum_mz.append(mz_float)
        spectrum_intensity.append(intensity_float)
        
        line = file_handle.readline()
        
    counter += 1
      
    peak_count = len(spectrum_mz)
    avg_peak_count += peak_count

    avg_peptide_len += peptide_len
    
    peptide_ids = [model_utils.vocab[x] for x in peptide]
    
    data_set.append([scan,
                     peptide_ids,
                     spectrum_mz,
                     spectrum_intensity,
                     peptide_mass,
                     charge])
   

  return data_set

  

def read_spectra_only_msp(file_handle, data_format, spectra_locations, artificial=False):
  
  data_set = []
  
  max_seq_len = model_utils.max_seq_len
  
  counter = 0
  counter_skipped = 0
  counter_skipped_mod = 0
  counter_skipped_len = 0
  counter_skipped_mass = 0
  counter_skipped_mass_precision = 0
  
  avg_peak_count = 0.0
  avg_peptide_len = 0.0

  if (data_format=="msp"):
    keyword="Name" 

  for location in spectra_locations:
    
    file_handle.seek(location)
    line = file_handle.readline()
    assert (keyword in line), "ERROR: read_spectra(); wrong format"
  
    unknown_modification = False
#    max_intensity = 0.0

    # READ AN ENTRY
    if (data_format=="msp"):

     # seq / charge
      raw_sequence, info = re.match("Name: (.*)/(.*)",line).group(1,2)
      
      mods = re.findall("(\d*\,\w\,\w*)\)",info)
      if len(mods)>0:
          added_len = 1
          for mod in mods:
              pos,AA,modtype = re.split("\,", mod)
              
              if modtype=="Acetyl":
                  raw_sequence = raw_sequence[:int(pos)+added_len] + "(+42.0106)" + raw_sequence[int(pos)+added_len:]
                  added_len += 10
              elif modtype=="Oxidation":
                  raw_sequence = raw_sequence[:int(pos)+added_len] + "(+15.99)" + raw_sequence[int(pos)+added_len:]
                  added_len += 8
                  
              elif modtype=="CAM":
                  raw_sequence = raw_sequence[:int(pos)+added_len] + "(+57.02)" + raw_sequence[int(pos)+added_len:]
                  added_len += 8
              elif modtype=="Amide":
                  raw_sequence = raw_sequence[:int(pos)+added_len] + "(+.98)" + raw_sequence[int(pos)+added_len:]
                  added_len += 6
              else:
                  raw_sequence = raw_sequence[:int(pos)+added_len] + "(+0)" + raw_sequence[int(pos)+added_len:]
                  added_len += 4
      # MW
      line = file_handle.readline()

      # Comment
      line = file_handle.readline()
      charge = int(re.match("\d",info).group(0))
      peptide_ion_mz = float(re.match(".*Parent=(\d*\.\d*)",line).group(1))
      
      # Num peaks
      line = file_handle.readline()
  
      raw_sequence_len = len(raw_sequence)
      peptide = []
      index = 0
      while (index < raw_sequence_len):
        if (raw_sequence[index]=="("):
          if (peptide[-1]=="C" and raw_sequence[index:index+8]=="(+57.02)"):
            peptide[-1] = "Cmod"
            index += 8
          elif (peptide[-1]=='M' and raw_sequence[index:index+8]=="(+15.99)"):
            peptide[-1] = 'Mmod'
            index += 8
          elif (peptide[-1]=='N' and raw_sequence[index:index+6]=="(+.98)"):
            peptide[-1] = 'Nmod'
            index += 6
          elif (peptide[-1]=='Q' and raw_sequence[index:index+6]=="(+.98)"):
            peptide[-1] = 'Qmod'
            index += 6
          else: 
            unknown_modification = True
            break
        else:
          peptide.append(raw_sequence[index])
          index += 1
      #
      if (unknown_modification):
        counter_skipped += 1
        counter_skipped_mod += 1
        continue
        
      peptide_mass = peptide_ion_mz*charge - charge*model_utils.mass_H -model_utils.mass_N_terminus-model_utils.mass_C_terminus
      
      if (peptide_mass > model_utils.MZ_MAX):
        counter_skipped += 1
        counter_skipped_mass += 1
        continue
    
      peptide_len = len(peptide)
      if (peptide_len > max_seq_len): 
        counter_skipped += 1
        counter_skipped_len += 1
        continue

      sequence_mass = sum(model_utils.mass_AA[aa] for aa in peptide)
      if (abs(peptide_mass-sequence_mass) > model_utils.PRECURSOR_MASS_PRECISION_INPUT_FILTER):
        counter_skipped_mass_precision += 1
        counter_skipped += 1
        continue

      spectrum_mz = []
      spectrum_intensity = []
      line = file_handle.readline()
      while(re.search("\t",line)):
        
        mz, intensity = re.split('\t', line)[:2]
        intensity_float = float(intensity)
        mz_float = float(mz)
        
        if (mz_float > model_utils.MZ_MAX): 
          line = file_handle.readline()
          continue
        
        spectrum_mz.append(mz_float)
        spectrum_intensity.append(intensity_float)
    
        line = file_handle.readline()

    counter += 1
    
    peak_count = len(spectrum_mz)
    avg_peak_count += peak_count

    avg_peptide_len += peptide_len
    
    peptide_ids = [model_utils.vocab[x] for x in peptide]
    
    data_set.append(["index:{0}".format(counter),
                     peptide_ids,
                     spectrum_mz,
                     spectrum_intensity,
                     peptide_mass])
   
  return data_set


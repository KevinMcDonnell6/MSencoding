
from pyteomics import mass
import numpy as np
import re 
import model_utils

atomic_mass = {"H":1.007825,
               "C":12.000000,
               "N":14.003074,
               "O":15.994915,
               "P":1.007276}

add =    {"a":-(atomic_mass["C"]+atomic_mass["O"]), 
           "b":0,         
           "c":atomic_mass["N"]+3*atomic_mass["H"],
           "x":atomic_mass["C"]+2*atomic_mass["O"],  
           "y":2*atomic_mass["H"]+atomic_mass["O"],    
           "z":-(atomic_mass["N"]+atomic_mass["H"])+atomic_mass["O"],
           "water":2*atomic_mass["H"]+atomic_mass["O"],
           "ammonia":atomic_mass["N"]+ 3*atomic_mass["H"]} 



def remove_mol(df,idx,mass,ion):
    df = df[:] 
    
    df_ion = [[df[j][i] for i in idx] for j in range(len(df))]
    
    df_ion[0] = [i-mass for i in df_ion[0]]
    df_ion[1] = [i+ion for i in df_ion[1]]
    for i in range(len(df)):
        df[i]+=df_ion[i]
    return df
    
def ionLoss(df,
            water=["Cterm", "D", "E", "S", "T"],
            ammonia=["K", "N", "Q", "R"]):
    
    assert len(df[0])==len(df[1])
    assert len(df[1])==len(df[2])
    
    rules = {"D":"^D.","E":"^E.","S":".S.","T":".T."}
    rules = [rules[i] for i in set(["D", "E", "S", "T"]).intersection(water)]
    wmatch = [re.match("|".join(rules),i) for i in df[2]]
    widx = [i for i in range(len(wmatch)) if wmatch[i] != None]
    CtermIdx = [i for i in range(len(df[1])) if df[1][i][0] in "xyz"]
    widx = list(set(widx + CtermIdx))

    rules = {"K":"^.*K.", "N":"^.*N.", "Q":"^.*Q.", "R":".R."}
    rules = [rules[i] for i in set(["K", "N", "Q", "R"]).intersection(ammonia)]
    amatch = [re.match("|".join(rules),i) for i in df[2]]
    aidx = [i for i in range(len(amatch)) if amatch[i] != None]

    df = remove_mol(df,widx,add["water"],ion="_")    
    df = remove_mol(df,aidx,add["ammonia"],ion="*")
    
    return df



# https://pyteomics.readthedocs.io/en/latest/examples/example_msms.html
def fragments(peptide, types=('b', 'y'), maxcharge=1, ion_loss=False):
    """
    The function generates all possible m/z for fragments of types 
    `types` and of charges from 1 to `maxharge`.
    """
    peptide = "".join(peptide)
    
    peaks = []
    ions = []
    frags = []
    pep_len = len(peptide)
    for i in range(1, pep_len):
        for ion_type in types:
            for charge in range(1, maxcharge+1):
                if ion_type[0] in 'abc':
                    peaks.append(mass.fast_mass(
                            peptide[:i], ion_type=ion_type, charge=charge))
                    ions.append(ion_type+str(i)+("+"+str(charge) if maxcharge>1 else ""))
                    frags.append(peptide[:i])
                else:
                    peaks.append(mass.fast_mass(
                            peptide[i:], ion_type=ion_type, charge=charge))
                    ions.append(ion_type+str(pep_len-i)+("+"+str(charge) if maxcharge>1 else ""))
                    frags.append(peptide[i:])
                    
    frags = [str(i) for i in frags]
    
    order = np.argpartition(ions,1)
    if maxcharge>1:
        order = sorted(range(len(ions)),key=lambda x: (ions[x][0],re.findall("(?:\d+)(\_|\*)*",ions[x])[0],int(re.search("(\d+)",ions[x])[0]),int(re.findall("(\d+)",ions[x])[1])))
    else:
        order = sorted(range(len(ions)),key=lambda x: (ions[x][0],re.findall("(?:\d+)(\_|\*)*",ions[x])[0],int(re.search("(\d+)",ions[x])[0])))
    peaks = list(np.array(peaks)[order])
    ions = list(np.array(ions)[order])
    frags = list(np.array(frags)[order])
     
    if ion_loss:
        df = ionLoss([peaks,ions,frags])
    else:
        df = [peaks,ions,frags]
    return df


pep_mass = mass.calculate_mass


def fragments_mgf(peptide, types=('b', 'y'), maxcharge=1, ion_loss=False):
    """
    The function generates all possible m/z for fragments of types 
    `types` and of charges from 1 to `maxharge`.
    """

    peaks = []
    ions = []
    frags = []
    pep_len = len(peptide)
    for i in range(1, pep_len):
        for ion_type in types:
            for charge in range(1, maxcharge+1):
                if ion_type[0] in 'abc':
                    peaks.append((sum([model_utils.mass_AA[j] for j in peptide[:i]])+
                                 model_utils.mass_C_terminus+
                                 model_utils.mass_N_terminus+
                                 (charge*model_utils.mass_H)+
                                 mass.calculate_mass(mass.std_ion_comp[ion_type]))/charge)
                    ions.append(ion_type+str(i)+("+"+str(charge) if maxcharge>1 else ""))
                    frags.append("".join(peptide[:i]))
                else:
                   peaks.append((sum([model_utils.mass_AA[j] for j in peptide[i:]])+
                                 model_utils.mass_C_terminus+
                                 model_utils.mass_N_terminus+
                                 (charge*model_utils.mass_H)+
                                 mass.calculate_mass(mass.std_ion_comp[ion_type]))/charge)
                   ions.append(ion_type+str(pep_len-i)+("+"+str(charge) if maxcharge>1 else ""))
                   frags.append("".join(peptide[i:]))
                    
    frags = [str(i) for i in frags]
    
    order = np.argpartition(ions,1)
    if maxcharge>1:
        order = sorted(range(len(ions)),key=lambda x: (ions[x][0],re.findall("(?:\d+)(\_|\*)*",ions[x])[0],int(re.search("(\d+)",ions[x])[0]),int(re.findall("(\d+)",ions[x])[1])))
    else:
        order = sorted(range(len(ions)),key=lambda x: (ions[x][0],re.findall("(?:\d+)(\_|\*)*",ions[x])[0],int(re.search("(\d+)",ions[x])[0])))
    
    peaks = list(np.array(peaks)[order])
    ions = list(np.array(ions)[order])
    frags = list(np.array(frags)[order])
    
    if ion_loss:
        df = ionLoss([peaks,ions,frags])
    else:
        df = [peaks,ions,frags]
    return df


def fragments_mgf_8ion(peptide, types=('b', 'y'), maxcharge=2, ion_loss=True):
    """
    The function generates all possible m/z for fragments of types 
    `types` and of charges from 1 to `maxharge`.
    """
    
    peaks = []
    ions = []
    frags = []
    pep_len = len(peptide)
    for i in range(1, pep_len):
        for ion_type in types:
            for charge in range(1, maxcharge+1):
                if ion_type[0] in 'abc':
                    peaks.append((sum([model_utils.mass_AA[j] for j in peptide[:i]])+
                                 model_utils.mass_C_terminus+
                                 model_utils.mass_N_terminus+
                                 (charge*model_utils.mass_H)+
                                 mass.calculate_mass(mass.std_ion_comp[ion_type]))/charge)
                    ions.append(ion_type+str(i)+("+"+str(charge) if maxcharge>1 else ""))
                    frags.append("".join(peptide[:i]))
                else:
                   peaks.append((sum([model_utils.mass_AA[j] for j in peptide[i:]])+
                                 model_utils.mass_C_terminus+
                                 model_utils.mass_N_terminus+
                                 (charge*model_utils.mass_H)+
                                 mass.calculate_mass(mass.std_ion_comp[ion_type]))/charge)
                   ions.append(ion_type+str(pep_len-i)+("+"+str(charge) if maxcharge>1 else ""))
                   frags.append("".join(peptide[i:]))
                    
    frags = [str(i) for i in frags]
    
    order = np.argpartition(ions,1)
    if maxcharge>1:
        order = sorted(range(len(ions)),key=lambda x: (ions[x][0],re.findall("(?:\d+)(\_|\*)*",ions[x])[0],int(re.search("(\d+)",ions[x])[0]),int(re.findall("(\d+)",ions[x])[1])))
    else:
        order = sorted(range(len(ions)),key=lambda x: (ions[x][0],re.findall("(?:\d+)(\_|\*)*",ions[x])[0],int(re.search("(\d+)",ions[x])[0])))
    
    peaks = list(np.array(peaks)[order])
    ions = list(np.array(ions)[order])
    frags = list(np.array(frags)[order])
    
    charge_2 = [peaks[1::2],ions[1::2],frags[1::2]]
    if ion_loss:
        df = ionLoss([peaks[::2],ions[::2],frags[::2]])
        for i in range(len(df)):
            df[i]+=charge_2[i]
    else:
        df = [peaks,ions,frags]
        
    return df


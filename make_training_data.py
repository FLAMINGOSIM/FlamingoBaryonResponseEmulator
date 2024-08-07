import numpy as np
from scipy import interpolate as inter
from matplotlib.pyplot import cm

h = 0.681

FLAMINGO_colors = ['#117733','#332288','#DDCC77','#CC6677','#abd0e6','#6aaed6','#3787c0','#105ba4','#FF8C40','#CC4314','#7EFF4B','#55E18E','#44AA99','#999933','#AA4499', '#882255']
FLAMINGO_labels = ["L1_m9","L2p8_m9","L1_m10","L1_m8","fgas+2sigma","fgas-2sigma","fgas-4sigma", "fgas-8sigma","M*-sigma","M*-sigma+fgas-4sigma","JETS","JETS fgas-4sigma","Planck","PlanckNu0p24Fix ","PlanckNu0p24Var", "LS8"]
FLAMINGO_plots = {name: color for name, color in zip(FLAMINGO_labels,FLAMINGO_colors)}

##
# @Return the file index corresponding to a given redshift
#
def index_from_z(z):
    if z > 2.0:
        print("Invalid range of z:", z)
        exit()
        
    return int(82 + (2.0 - z) / 0.05)

def model_from_props(sigma_gas, sigma_star, jet):

    if sigma_star == -1:
        if jet != 0.:
            print("Cannot combine jet and sigma_star!")
        
        if sigma_gas == 0:
            return "HYDRO_STRONG_SUPERNOVA"
        elif sigma_gas == -4:
            return "HYDRO_STRONGER_AGN_STRONG_SUPERNOVA"
        else:
            print("Invalid sigma_gas for non-zero sigma_star:", sigma_gas)
            exit()            
    elif sigma_star != 0.:
        print("Invalid sigma_star:", sigma_star)
        exit()

    # From here: sigma_star == 0.
    
    if jet:
        if sigma_gas == -4:
            return "HYDRO_STRONG_JETS"
        elif sigma_gas == 0:
            return "HYDRO_JETS"
        else:
            print("Invalid sigma_gas for jets:", sigma_gas)
            exit()
    else:        
        if sigma_gas == -8:
            return "HYDRO_STRONGEST_AGN"
        elif sigma_gas == -4:
            return "HYDRO_STRONGER_AGN"
        elif sigma_gas == -2:
            return "HYDRO_STRONG_AGN"
        elif sigma_gas == 0:
            return "HYDRO_FIDUCIAL"
        elif sigma_gas == 2:
            return "HYDRO_WEAK_AGN"
        else:
            print("Invalid sigma_gas:", s)
            exit()

def get_index_flamingo_arrays(sigma_gas, sigma_star, jet):

    if sigma_star == -1:
        if jet != 0.:
            print("Cannot combine jet and sigma_star!")
        
        if sigma_gas == 0:
            return 8
        elif sigma_gas == -4:
            return 9
        else:
            print("Invalid sigma_gas for non-zero sigma_star:", sigma_gas)
            exit()            
    elif sigma_star != 0.:
        print("Invalid sigma_star:", sigma_star)
        exit()

    # From here: sigma_star == 0.
    if jet:
        if sigma_gas == -4:
            return 11
        elif sigma_gas == 0:
            return 10
        else:
            print("Invalid sigma_gas for jets:", sigma_gas)
            exit()
    else:        

        if sigma_gas == -8:
            return 7
        elif sigma_gas == -4:
            return 6
        elif sigma_gas == -2:
            return 5
        elif sigma_gas == 0:
            return 0
        elif sigma_gas == 2:
            return 4
        else:
            print("Invalid sigma_gas:", sigma_gas)
            exit()
    

        
##
# @ Return the PS ratio for a given model and redshift
# at a given k [h / Mpc] by interpolating nearby bins
#
def PS_ratio(k, z, sigma_gas, sigma_star, jet, fix_low_k_norm):
    
    index = index_from_z(z)
    model = model_from_props(sigma_gas, sigma_star, jet)
    fname = "../data/%s/ratio_%0.4d.txt"%(model, index)

    # Correct data for h
    data = np.loadtxt(fname)
    k_data = data[:,0] / h 
    P_data = data[:,1]

    # Build linear interpolator 
    #interpolator = inter.interp1d(np.log10(k_data), P_data)
    interpolator = inter.CubicSpline(np.log10(k_data), P_data)

    if fix_low_k_norm and jet == 1.:
        offset = (0.99985 + 0.000034*z - interpolator(-2.))
        P_data += offset
        interpolator = inter.CubicSpline(np.log10(k_data), P_data)

    #print("model:", model, "z=", z, "R(-inf)=", P_data[0])

    return interpolator(np.log10(k))

###########################################

def make_training_data(z_train, model_train, k_min, k_max, num_bins_k, fix_low_k_norm=True, rand_k_norm=0.):

    bins_k = []
    bins_R = []
    labels = []
    color_model = []
    color_z = []
    sigmas_gas = []
    sigmas_star = []
    redshifts = []
    jets = []

    colors_z = cm.plasma(np.linspace(0., 0.9, len(z_train)))
    
    for i in range(len(model_train)):
        for j in range(len(z_train)):
            sigma_gas = model_train[i][0]
            sigma_star = model_train[i][1]
            jet = model_train[i][2]
            z = z_train[j]
            
            index_flamingo = get_index_flamingo_arrays(sigma_gas, sigma_star, jet)
            
            my_k = 10**(np.linspace(np.log10(k_min), np.log10(k_max), num_bins_k) + np.random.random(num_bins_k) * rand_k_norm - rand_k_norm / 2. )
            my_R = PS_ratio(my_k, z, sigma_gas, sigma_star, jet, fix_low_k_norm)
            bins_k.append(my_k)
            bins_R.append(my_R)
            color_model.append(FLAMINGO_colors[index_flamingo])
            color_z.append(colors_z[j])
            labels.append(FLAMINGO_labels[index_flamingo])
            sigmas_gas.append(sigma_gas)
            sigmas_star.append(sigma_star)
            redshifts.append(z)
            jets.append(jet)
            
    return bins_k, bins_R, labels, color_model, color_z, sigmas_gas, sigmas_star, jets, redshifts
    

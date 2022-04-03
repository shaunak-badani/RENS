class Units:
    kB = 1   
    epsilon = 1

    # flag to differentiate between arbitrary and standard units
    arbitrary = True

    # FOR LJ system : 
    BOLTZMANN = 1.380649 * 1e-23 # J / K
    NA = 6.023 * 1e23
    KILO = 1e3
    RGAS = BOLTZMANN * NA 
    BOLTZ = RGAS / KILO # kB in kJ / mol K

    AMU_TO_KG = 1 / (KILO * NA) 
    A_PS_TO_M_S = 1e2
    M_S_TO_A_PS = 1 / (A_PS_TO_M_S)
    kJ_mol_TO_J = KILO / NA
    S_TO_PS = 1e-12
    


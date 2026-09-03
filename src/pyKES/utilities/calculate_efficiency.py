from pyKES.utilities.unit_handler import Quantity

PLANCK_CONSTANT = Quantity(6.62607015e-34, "J*s")  # Planck's constant in Joule-seconds
SPEED_OF_LIGHT = Quantity(2.99792458e8, "m/s")  # Speed of light in meters per second
AVOGADRO_NUMBER = Quantity(6.02214076e23, "1/mol")  # Avogadro's number in 1/mol

def calculate_apparent_quantum_yield(irradiation_wavelength: Quantity,
                                     irradiation_area: Quantity,
                                     irradiance_power: Quantity,
                                     reaction_rate: Quantity,
                                     fraction_of_photons_reaching_inside: Quantity,
                                     electron_transfer_per_reaction: int = 4
                                     ) -> Quantity:
    '''
    Calculate the apparent quantum yield (AQY) of a photochemical reaction.
    
    Parameters
    ----------
    irradiation_wavelength : Quantity
        The wavelength of the incident light (Quantity, length)
    irradiation_area : Quantity
        The area of the irradiated surface (Quantity, area)
    irradiance_power : Quantity
        The power of the incident light (Quantity, power / area)
    reaction_rate : Quantity
        The rate of the reaction (Quantity, substance/time)
    fraction_of_photons_reaching_inside : Quantity
        The fraction of incident photons that reach the inside of the reactor (Quantity, dimensionless)
    electron_transfer_per_reaction : int, optional
        The number of electrons transferred per reaction (e.g. 4 when reaction rate is for O2 production,
        2 when reaction rate is for H2 production), by default 4

    Returns
    -------
    Quantity
        The apparent quantum yield (AQY) of the photochemical reaction.
    '''

    photon_energy = Quantity((PLANCK_CONSTANT.unit['J * s'] 
                              * SPEED_OF_LIGHT.unit['m / s']) 
                              / irradiation_wavelength.unit['m'],
                              'J')

    incident_power = Quantity(irradiance_power.unit['W / m2']
                              * irradiation_area.unit['m2'],
                              'W')

    incident_photon_flux = Quantity(incident_power.unit['W']
                                    / photon_energy.unit['J'],
                                    '1/s')

    effective_photon_flux = Quantity(incident_photon_flux.unit['1/s']
                                     * fraction_of_photons_reaching_inside.unit['-'],
                                     '1/s')

    molecules_per_time = Quantity(reaction_rate.unit['mol / s']
                                   * AVOGADRO_NUMBER.unit['1 / mol'],
                                     '1/s')

    apparent_quantum_yield = Quantity(electron_transfer_per_reaction *  # e.g. 4 electrons are required to produce one molecule of O2
                                      molecules_per_time.unit['1 / s']
                                      / effective_photon_flux.unit['1 / s'],
                                      '-')

    return apparent_quantum_yield

    
def light_to_hydrogen_efficiency(irradiation_area: Quantity,
                                 irradiance_power: Quantity,
                                 reaction_rate: Quantity,
                                 electron_transfer_per_reaction: int = 4
                                 ) -> Quantity:
    '''
    Calculate the light-to-hydrogen efficiency of a photochemical water-splitting reaction.
    This calculation is simply based on incident power, thus the spectral distribution of the 
    light source is not considered (if incident light is solar light, this calculation is equivalent
    to solar-to-hydrogen efficiency).

    Parameters
    ----------
    irradiation_area : Quantity
        The area of the irradiated surface (Quantity, area)
    irradiance_power : Quantity
        The power of the incident light (Quantity, power / area)
    reaction_rate : Quantity
        The rate of the reaction (Quantity, substance/time)
    electron_transfer_per_reaction : int, optional
        The number of electrons transferred per reaction (e.g. 4 when reaction rate is for O2 production,
        2 when reaction rate is for H2 production), by default 4   
    
    Returns
    -------
    Quantity
        The light-to-hydrogen efficiency of the photochemical water-splitting reaction 
        (Quantity, dimensionless).

    '''
    
    WATER_SPLITTING_GIBBS_ENERGY = Quantity(474.48, 'kJ /mol') # for 2 H2O -> 2 H2 + O2, per mole of O2 produced
    WATER_SPLITTING_GIBBS_ENERGY_PER_ELECTRON = Quantity(WATER_SPLITTING_GIBBS_ENERGY.unit['kJ / mol']
                                                         / 4, # 4 electrons are required to produce one molecule of O2
                                                         'kJ / mol')
    
    incident_power = Quantity(irradiance_power.unit['W / m2']
                              * irradiation_area.unit['m2'],
                              'W')

    energy_converted_per_time = Quantity(reaction_rate.unit['mol / s'] 
                                         * electron_transfer_per_reaction
                                         * WATER_SPLITTING_GIBBS_ENERGY_PER_ELECTRON.unit['J / mol'],
                                         'J /s')

    light_to_hydrogen_efficiency = Quantity(energy_converted_per_time.unit['J / s']
                                            / incident_power.unit['W'],
                                            '-')

    return light_to_hydrogen_efficiency


def testing():

    aqy = calculate_apparent_quantum_yield(
        irradiation_wavelength = Quantity(365, "nm"),
        irradiation_area = Quantity(2.5, "cm2"),
        irradiance_power = Quantity(44, "mW/cm2"),
        reaction_rate = Quantity(5.2e-5, "mol[O2]/h"),
        fraction_of_photons_reaching_inside = Quantity(0.86, "-")
    )

    print(f"Apparent Quantum Yield: {aqy.unit['%']:.2f}%")

    lth = light_to_hydrogen_efficiency(
        irradiation_area = Quantity(320, "cm2"),
        irradiance_power = Quantity(40, "mW/cm2"),
        reaction_rate = Quantity(5.84e-8, "mol[H2]/s"),
        electron_transfer_per_reaction = 2
    )

    print(f"Light to Hydrogen Efficiency: {lth.unit['%']:.4f}%")

if __name__ == "__main__":
    testing()
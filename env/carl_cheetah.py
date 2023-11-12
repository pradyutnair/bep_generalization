from carl.envs import CARLBraxHalfcheetah


def carl_cheetah_environment(scaling_factor: int = 2.75):
    """
    Creates a CARLBraxHalfcheetah environment with the context space and sampler
    :param scaling_factor: 
    :return: 
    """
    context_dict = {'gravity': -9.8,
                    'friction': 1.0,
                    'elasticity': 0.0,
                    'ang_damping': -0.05,
                    'viscosity': 0.0,
                    'mass_torso': 10.0,
                    'mass_bthigh': 1.5435146,
                    'mass_bshin': 1.5874476,
                    'mass_bfoot': 1.0953975,
                    'mass_fthigh': 1.4380753,
                    'mass_fshin': 1.2008368,
                    'mass_ffoot': 0.8845188}
    # Multiply all values in the dictionary by scaling factor
    context_dict = {k: v * scaling_factor for k, v in context_dict.items() if k != 'gravity'}
    context_dict['gravity'] = -9.8
    modified_cheetah = CARLBraxHalfcheetah(context=context_dict)
    return modified_cheetah

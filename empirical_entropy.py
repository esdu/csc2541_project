import numpy as np

def empirical_entropy(samples, num_bins=80):
    '''Assumption: Input values are between 0 and 6 (the ratings)'''
    # histogram method
    # put variables into 6/(num_bins)-width bins from 0 to 5
    bounds = np.linspace(-20,20,num_bins+1)
    frequencies = np.zeros((num_bins, samples.shape[1]))
    num_samples = samples.shape[0]
    for i in range(num_bins):
        indicators = samples*(samples <= bounds[i+1]) > bounds[i]
        frequencies[i] = np.sum(indicators, axis=0)

    # toget rid of 0s when taking log
    mock_frequencies = frequencies + (frequencies==0)


    return frequencies, -np.sum((frequencies/num_samples)*(np.log(mock_frequencies/(num_samples))), axis=0)


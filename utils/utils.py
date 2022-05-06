import numpy as np

def calculate_ma(scores, ma_period):
    moving_average = np.zeros(len(scores))
    for i in range(len(moving_average)):
        moving_average[i] = np.mean(scores[max(0, i - ma_period):(i+1)])
    
    return moving_average
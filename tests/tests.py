from commons.interactive_plots import commission_analysis
from commons.write_results import read_simple_rewards_commissions
import numpy as np
import tensorflow as tf



input = tf.placeholder("float", [1, 3])
action = tf.placeholder("float", [1, 1])

# Action


init = tf.initialize_all_variables()
with tf.Session() as session:

    session.run(init)

    x1_data = np.zeros((1, 3))

    x1_data[0][0] = 1
    x1_data[0][1] = 2
    x1_data[0][2] = 3

    a_data = np.zeros((1,1))

    '''
    x2_data = np.zeros((3, 1))
    
    x2_data[0][0] = 1
    x2_data[1][0] = 2
    x2_data[2][0] = 3
    

    result = session.run(y, feed_dict={input: x1_data,
                                       action: a_data
                                       })
    '''

    #print(result)

'''
decission_changes_02 = read_simple_rewards_commissions('0.1')
decission_changes_05 = read_simple_rewards_commissions('0.5')
decission_changes_1 = read_simple_rewards_commissions('1')

commission_analysis(decission_changes_02, decission_changes_05, decission_changes_1)
'''
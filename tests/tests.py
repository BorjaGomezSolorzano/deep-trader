from commons.interactive_plots import commission_analysis
from commons.write_results import read_simple_rewards_commissions

decission_changes_02 = read_simple_rewards_commissions('0.1')
decission_changes_05 = read_simple_rewards_commissions('0.5')
decission_changes_1 = read_simple_rewards_commissions('1')

commission_analysis(decission_changes_02, decission_changes_05, decission_changes_1)
from commons.interactive_plots import commission_analysis
from commons.write_results import read_simple_rewards_commissions

decission_changes_02 = read_simple_rewards_commissions('2.5e-05')
decission_changes_05 = read_simple_rewards_commissions('0.000125')
decission_changes_1 = read_simple_rewards_commissions('0.00025')

commission_analysis(decission_changes_02, decission_changes_05, decission_changes_1)
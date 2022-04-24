from lumo import Experiment

exp = Experiment('my_first_try')
logstd = exp.test_file('stdout.log','log')
logerr = exp.test_file('stderr.log','log')
print(logstd)
print(logerr)
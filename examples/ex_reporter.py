"""

"""

from thexp.frame.drawer import Reporter

reporter = Reporter("./experiment")

for i in range(100):
    reporter.add_scalar(i**2,i,"tw")
    reporter.add_scalar(i**3,i,"twa")

reporter.savefig()
reporter.savearr()

for i in range(100):
    reporter.add_scalar(i ** 2, i, "tw")
    reporter.add_scalar(i ** 3, i, "twa")

reporter.report()
import numpy as np
import chaospy

distribution = chaospy.Uniform(0, 20)
samples = distribution.sample(5, rule="sobol")
evaluations = samples**2
print(samples)
print(evaluations)

from matplotlib import pyplot

expansion = chaospy.generate_expansion(10, distribution, normed=True)

t = np.linspace(0, 20, 10)
# pyplot.rc("figure", figsize=[15, 6])
# pyplot.plot(t, expansion(t).T)
# pyplot.plot(samples, evaluations, ".", markersize=10)
# # pyplot.axis([0, 15, -3, 3])
# pyplot.show()

from sklearn.linear_model import LarsCV

lars = LarsCV(fit_intercept=False, max_iter=5)
pce, coeffs = chaospy.fit_regression(
    expansion, samples, evaluations, model=lars, retall=True)
expansion_ = expansion[coeffs != 0]

pce.round(2)

lars = LarsCV(fit_intercept=False, max_iter=5)
lars.fit(expansion(samples).T, evaluations)
expansion_ = expansion[lars.coef_ != 0]

lars.coef_.round(4)

print("number of expansion terms total:", len(expansion))
print("number of expansion terms included:", len(expansion_))

import gstools

model = gstools.Gaussian(dim=1, var=1)
pck = gstools.krige.Universal(model, samples, evaluations, list(expansion_))

pck(samples)
assert np.allclose(pck.field, evaluations)

uk = gstools.krige.Universal(model, samples, evaluations, "linear")
uk(samples)
assert np.allclose(uk.field, evaluations)

pck(t)
mu, sigma = pck.field, np.sqrt(pck.krige_var)
pyplot.plot(t, mu, label="pck")
pyplot.fill_between(t, mu-sigma, mu+sigma, alpha=0.4)

uk(t)
mu, sigma = uk.field, np.sqrt(uk.krige_var)
pyplot.plot(t, mu, label="uk")
pyplot.fill_between(t, mu-sigma, mu+sigma, alpha=0.4)

pyplot.plot(t, pce(t), label="pce")

pyplot.scatter(samples, evaluations, color="k", label="samples")

# pyplot.axis([0, 15, -12, 15])
pyplot.legend(loc="upper left")
pyplot.show()


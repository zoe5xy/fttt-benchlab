# Third-party notices

FTTT BenchLab contains the source needed to run DIP, INR and DPS. No separate
checkout or installation of the source projects below is required. Standard
Python dependencies are declared in pyproject.toml.

## Neural models and solvers

The model and solver implementations were adapted from
https://github.com/peterw634/inv_framework at commit
b258fa90d284f0105259aeffbfd43814f4cd3085. The upstream README declares the MIT
license; that revision does not contain a separate license file or an explicit
copyright-holder statement. This notice preserves the published declaration
without inventing an additional copyright attribution or license grant.
Original file identifiers are recorded in SOURCE.json.

Adaptations include FTTT-specific observations and physical-variable mapping,
loss/observer/budget hooks, discrete Eikonal autograd, bounded reconstruction,
and a reverse schedule with unique time steps and a clean final sample.

## Eikonal and benchmark infrastructure

The Eikonal implementation and benchmark support include MIT-licensed code from
USCT BenchLab. Its copyright and permission notice are retained in LICENSE.
These modules are included directly in this package and import only this package
and the declared Python dependencies.

## Synthetic reference prior

The supplied 16x16 synthetic-ellipse checkpoint is an example prior, with its
training and validation metadata and SHA256 included. It is not a trained prior
for experimental breast data. The complete training entry point is
ftttbench.algorithms.fttt.train_prior.

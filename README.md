# M4_NN_0
try

The MAPE scores have a relatively big variation. Before this repo, I have written a test code and in that project the errors were substantially  lower.
I am not sure what could cause such a problem. It seems to me, there is a mistake around scaling and inverse transforming the results. 

I was expecting 4 to 8% errors, but sometimes this expectation does not hold. 

I am also adding the already split csv files, so probing could be done easily on a few data files before actually training on 100.000 files.

I am using new sklearn, to install it there is a file included named 'scikit_learn-0.24.dev0-cp38-cp38-macosx_10_9_x86_64.whl'
You can write;
pip install scikit_learn-0.24.dev0-cp38-cp38-macosx_10_9_x86_64.whl inside of the project.

Then it should work. The new sklearn for calculating MAPE.

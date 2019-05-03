fprintf('My approach to getting the best possible accuracy was to keep my cell size as small as possible to make sure I wasnt \n'); 
fprintf('losing any data form the originl image. As for the regularization parameter lambda for the SVM algorithm, I tried to also \n'); 
fprintf('keep it an even value. Since the lambda parameter assigns importance to wrongful classifications, it was a bit tricky');
fprintf('figuring out a value that would not be underfitting or overfitting the model ');
fprintf('\n');
fprintf('The lambda value seemed to affect the accuracy of the test test a great deal.\n'); 
fprintf('I noticed that it would increase the accruacy when I decreased the value for lambda, but only to a certain degree \n'); 
fprintf('before it wouldnt improve it any further. Therefore I chose my value of lambda to produce to best and closest accuracy \n')
fprintf('at a value of 0.1 as seen by my my_svm.mat MATLAB file \n');
fprintf('\n');
fprintf('This same inverse relationship was true about the cellSize chosen. Overall I went with a cellsize of 6 as it produced \n');
fprintf('the closest percentage accuracy for my validation set at 99.2 percent accurate \n');

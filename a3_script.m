%CP 8307 Introduction To Computer Vision%
%Assignment 3 - Machine Learning & Optical Flow Estimation%
%Name: Pouya Khosravi%
%April 20, 2019%

%-------------------------------------------------------------------------%

%Part 1 - Face Detection 

%-------------------------------------------------------------------------%

% 1. [5 points]Using the images inimagesnotfaces, generate a set of cropped, 
%grayscale, non-face images. Usegeneratecroppednotfaces.mas your starting 
%point. The images should be 3636, like the face images

% you might want to have as many negative examples as positive examples

d = 'C:\Users\Pouya\Documents\MATLAB\images_notfaces';
mkdir('cropped_training_images_notfaces'); 
S = dir(fullfile(d,'*.jpg')); % pattern to match filenames
total_images = numel(S);    % count total number of photos present in that folder
dim = 36;

for n = 1:total_images
    full_name= fullfile(d, S(n).name);         % it will specify images names with full path and extension
    our_images = imread(full_name);                                       % Read images  
    Gray  = rgb2gray(our_images);
    GrayS = imresize(Gray, [dim, dim], 'bilinear'); 
    baseFileName = "notface" + n + ".jpg"; 
    fullFileName = fullfile('cropped_training_images_notfaces' , baseFileName);
    imwrite(GrayS, fullFileName);
end

pause();

% 2. [5 points]Split your training images into two sets: a training set, and 
%a validation set. A good rule of thumb isto use 80% of your data for 
%training, and 20% for validation

d2 = 'C:\Users\Pouya\Documents\MATLAB\cropped_training_images_notfaces\';
S2 = dir(fullfile(d2,'*.jpg'));

[m,n] = size(S2) ;
P = 0.80 ;
idx = randperm(m)  ;


Training = S2(idx(1:round(P*m)),:) ; 
total_training_images = numel(Training);

for i = 1:total_training_images
    training_names= fullfile(d2, Training(i).name);         % it will specify images names with full path and extension
    our_training_images = imread(training_names);
    baseFileName = "notface" + "train" + i + ".jpg"; 
    fullFileName = fullfile('cropped_training_images_notfaces\' , baseFileName);
    imwrite(our_training_images, fullFileName);
end
    
Validation = S2(idx(round(P*m)+1:end),:) ;
total_validating_images = numel(Validation);

for i = 1:total_validating_images
    validating_names= fullfile(d2, Validation(i).name);         % it will specify images names with full path and extension
    our_validating_images = imread(validating_names);
    baseFileName = "notface" + "validation" + i + ".jpg"; 
    fullFileName = fullfile('cropped_training_images_notfaces\' , baseFileName);
    imwrite(our_validating_images, fullFileName);
end

for n = 1:total_images
    full_name= fullfile(d, S(n).name);         % it will specify images names with full path and extension
    our_images = imread(full_name);
    baseFileName = "notface" + n + ".jpg"; 
    fullFileName = fullfile('cropped_training_images_notfaces\' , baseFileName);
    delete(fullFileName);
end

pause();

% 3. [5 points]Generate HOG features for all of your training and validation 
%images. Use getfeatures.m as your starting point.  You are free to experiment
%with the details of your HOG descriptior.

pos_imageDir = 'cropped_training_images_faces';
pos_imageList = dir(sprintf('%s/*.jpg',pos_imageDir));
pos_nImages = length(pos_imageList);

neg_imageDir = 'cropped_training_images_notfaces';

neg_imageTrainList = dir(sprintf('%s/notfacetrain*.jpg',neg_imageDir));
neg_imageValList = dir(sprintf('%s/notfacevalidation*.jpg',neg_imageDir));

neg_nTrainImages = length(neg_imageTrainList);
neg_nValImages = length(neg_imageValList);

cellSize = 6;
featSize = 31*cellSize^2;
    
pos_feats = zeros(pos_nImages,featSize);
for i=1:pos_nImages
    im = im2single(imread(sprintf('%s/%s',pos_imageDir,pos_imageList(i).name)));
    feat = vl_hog(im,cellSize);
    pos_feats(i,:) = feat(:);
    fprintf('got feat for pos image %d/%d\n',i,pos_nImages);
%     imhog = vl_hog('render', feat);
%     subplot(1,2,1);
%     imshow(im);
%     subplot(1,2,2);
%     imshow(imhog)
%     pause;
end

neg_feats_train = zeros(neg_nTrainImages,featSize);
for i=1:neg_nTrainImages
    im = im2single(imread(sprintf('%s/%s',neg_imageDir,neg_imageTrainList(i).name)));
    feat = vl_hog(im,cellSize);
    neg_feats_train(i,:) = feat(:);
    fprintf('got feat for neg train image %d/%d\n',i,neg_nTrainImages);
%     imhog = vl_hog('render', feat);
%     subplot(1,2,1);
%     imshow(im);
%     subplot(1,2,2);
%     imshow(imhog)
%     pause;
end


neg_feats_val = zeros(neg_nValImages,featSize);
for i=1:neg_nValImages
    im = im2single(imread(sprintf('%s/%s',neg_imageDir,neg_imageValList(i).name)));
    feat = vl_hog(im,cellSize);
    neg_feats_val(i,:) = feat(:);
    fprintf('got feat for neg validation image %d/%d\n',i,neg_nValImages);
%     imhog = vl_hog('render', feat);
%     subplot(1,2,1);
%     imshow(im);
%     subplot(1,2,2);
%     imshow(imhog)
%     pause;
end

save('pos_neg_feats.mat','pos_feats','neg_feats_train','neg_feats_val','pos_nImages','neg_nTrainImages','neg_nValImages')

pause();

% 4. [5 points] Train an SVM on the features from your training set.
%Usetrainsvm.m as your starting point.The parameter “lambda” will help you 
%control overfitting

load('pos_neg_feats.mat')

feats = cat(1,pos_feats,neg_feats_train);
labels = cat(1,ones(pos_nImages,1),-1*ones(neg_nTrainImages,1));

lambda = 0.1;
[w,b] = vl_svmtrain(feats',labels',lambda);

fprintf('Classifier performance on train data:\n')
confidences = [pos_feats; neg_feats_train]*w + b;

[tp_rate, fp_rate, tn_rate, fn_rate] =  report_accuracy(confidences, labels);


%5.[5 points] Test your SVM on the validation set features. From the SVM’s
%performance at this step, try to refine the parameters you chose in the 
%earlier steps (e.g., the cell size for HOG, and lambda for the SVM). Save
%your final SVM (weights and bias) in amatfile calledmysvm.mat, and include
%it in your submission.

load('pos_neg_feats.mat')

feats = cat(1,pos_feats,neg_feats_val);
labels = cat(1,ones(pos_nImages,1),-1*ones(neg_nValImages,1));

lambda = 0.1;
[w,b] = vl_svmtrain(feats',labels',lambda);

fprintf('Classifier performance on Validation data:\n')
confidences = [pos_feats; neg_feats_val]*w + b;

[tp_rate, fp_rate, tn_rate, fn_rate] =  report_accuracy(confidences, labels);

save('my_svm.mat','lambda','cellSize');   %Saving the variable vlaues chosen for lambda and cell size in a custom mat file

pause();

%-------------------------------------------------------------------------%

%Part 2 - Optical Flow Estimation 

%-------------------------------------------------------------------------%

%Write a MATLAB function, call itmyFlow, that takes as input two images,
%img1 andimg2, the window lengthused to compute flow around a point, and a
%threshold. The function should return three images,uandv, thatcontain 
%the horizontal and vertical components of the estimated optical flow, 
%respectively, and a binary map thatindicates whether the flow is valid.

fprintf('When going through trail and error, I noticed that A small window size makes our algorithm detect subtle changes in  motion but excludes larger ranges of motion.\n')
fprintf('Increasing the set threshold value will ultimately decrease the algorithm sensitivity, making it less seseptible to only non-large movement, but larger window sizes do the opposite.\n');
fprintf('Based on our known theory about the K-T feature tracker, this algorithm only performs well if the motion between frames is small.\n')

s_dir = 'C:\Users\Pouya\Documents\MATLAB\Sequences';
s_list = dir(fullfile(s_dir,'*.png'));
n_seq = length(s_list);

counter = 1; %counting the number of png files we have in our Sequenceing files 

%Part 2. Plotting the motion on color using flowToColor function.

for idz = 1:n_seq/2
    
    img_path = s_list(counter).folder + "\" + s_list(counter).name;
    counter = counter + 1;
    
    img1 = imread(img_path);
    
    img_path = s_list(counter).folder + "\" + s_list(counter).name;
    counter = counter + 1;
    img2 = imread(img_path);

    img1 = imresize(img1,0.8);    %Resizing our first image of the three                                   
    img2 = imresize(img2,0.8);    %Resizing our second image of the three

    counter2 = 1;                  

    ww = [5, 15, 45];                        % Setting our vairous window sizes 
    threshold_values = [0.001, 0.005, 0.01];  % Setting our various thrshold values 

    figure(idz);
    
    sgtitle(s_list(counter-1).name)
    for idy = 1:numel(threshold_values)    
        
        threshold = threshold_values(idy);
        
        for idx = 1:numel(ww)
            ww_size = ww(idx);
            flow = [];
            [flow(:,:,1),flow(:,:,2)] = myFlow(img1,img2, ww,threshold);   
            imageFlow = flowToColor(flow);      %Calling our flow to colour MATLAB function  
            
            subplot(numel(threshold_values),numel(ww),counter2), 
            imshow(imageFlow)
            
            title("Map for W-Size="+ww_size+" & t="+threshold) 
            counter2 = counter2 + 1;
        end
    end
end

%Part 1 -  Creating our myFlow function to estimates the motion between frames

function [u,v] = myFlow(img1,img2,w_size, threshold)

    w = floor(w_size/2);                        %establishing our window sizes for our myFlow function
    
    gauss_filter = (1/12)*[-1 8 0 -8 1];        %using the gaussian fiulter provided in the question 
    
    gausssian_dx = fliplr(gauss_filter);     %Calculating our Gaussian derivative in the x direction     
    gausssian_dy = (fliplr(gauss_filter))';  %Calculating our Gaussian derivative in the x direction

    img1 = im2double(img1);                  %Converting both of our images into 'double' formats 
    img2 = im2double(img2);
        
    if size(img1,3) == 3                    %loop statement to convert our images from RGB format to grayscale
        img1 = single(rgb2gray(img1));
    end
    if size(img2,3) == 3
        img2 = single(rgb2gray(img2));
    end

    dim_img1 = size(img1);

    %Apply filter on entire image 1 on the x axis and the y axis.
    
    img1_dx = conv2(img1, gausssian_dx);
    img1_dx = imresize(img1_dx, dim_img1);                  % Resizing our image to its orignal size.
    img1_dy = conv2(img1, gausssian_dy);
    img1_dy = imresize(img1_dy, dim_img1);                  % Resizing our image to its orignal size.


    img_dt = imgaussfilt(img1,1, 'FilterSize',3) - imgaussfilt(img2,1,'FilterSize',3); %Setting our gaussian filter size to a 3x3 filter as recommended by the question

    u = zeros(dim_img1);  %initializing our x direction changes 
    v = zeros(dim_img1);  %initializing our y direction changes 

    for i = w:dim_img1(2)- w -1                         %Test to see if the pixel range chosen is solvable.
       
        for j = w:dim_img1(1)- w -1
            
                Ix = img1_dx(i-w+1:i+w+1, j-w+1:j+w+1);
                Ix = reshape(Ix,1,[]);
                Iy = img1_dy(i-w+1:i+w+1, j-w+1:j+w+1);
                Iy = reshape(Iy,1,[]);
                It = img_dt(i-w+1:i+w+1, j-w+1:j+w+1);
                It = reshape(It,1,[]);
            
                sym_eig = [ sum(Ix.^2) , sum(Ix.*Iy) ;
                            sum(Ix.*Iy), sum(Iy.^2) ];
                eig_values = eig(sym_eig);
            
                %If any eigen  values < threshold it is not symmetric and optical flow cannot be calculated.
                
                if any(eig_values < threshold)
                    u(i,j) = 0;
                    v(i,j) = 0;
                else
                    S(:,1) = Ix';
                    S(:,2) = Iy';
                    
                    %if it has positive eigen values we can calculate the movement using least squares
                    
                    mov = inv(S'*S)*S'*It';
                    u(i,j) = mov(1);
                    v(i,j) = mov(2);
                end
            
        end
    end
end


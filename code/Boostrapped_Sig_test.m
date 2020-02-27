% select the variable  - All that start with bin are the binned ones (self
% explanatory )
clear MeanFitResultsAI;
% Field Trial
% youngdata = Bin_FF_PE_MaxSigned (Young,:);
% olddata = Bin_FF_PE_MaxSigned (Old,:);
a = 1;

% %%% Channel Trial
% youngdata = MeanArray_CH_AdaptIndex (Young,17:end);
% olddata = MeanArray_CH_AdaptIndex (Old,17:end);
% a = 2;

BootSamps = 1000;
Yboot{1} = bootstrp( BootSamps ,@mean,Control_RAW);
Yboot{2} = bootstrp( BootSamps ,@mean,Expert_RAW);

for group = 1:2
    for n = 1: BootSamps

        y = Yboot{group} (n,:)';
        x = (1:length (y))';

      if a == 2
        g                             = fittype('-a*exp((-b)*x)+c','dependent',{'y'},...
                                        'independent',{'x'}, 'coefficients',{'a','b','c'});
        [aa,goodness,details]         = fit(x,y,g,'Lower',[0 0 0],'StartPoint',[0.1 -0.05 1]); %3 was -10
        MeanFitResultsAI{group} (n,:) = [aa.a,aa.b,aa.c,goodness.adjrsquare];
      else
        g                             = fittype('a*exp((-b)*x)+c','dependent',{'y'}...
                                       ,'independent',{'x'}, 'coefficients',{'a','b','c'});
        [aa,goodness,details]         = fit(x,y,g,'Lower',[0 0 0],'StartPoint',[5 0.05 1]); %3 was -10
        MeanFitResultsAI{group} (n,:) = [aa.a,aa.b,aa.c,goodness.adjrsquare];
      end
        
%         gFIT                    = fit(x,y,g,'Lower',[0 0 0],'StartPoint',[0.1 0.5 1]);
%         figure(n)
%         hold on;
%         plot(gFIT,x,y);
    end
end
figure

title ('y constant'), hold on

histogram(MeanFitResultsAI{1}(:,1),'facecolor','g'); hold on

histogram(MeanFitResultsAI{2}(:,1),'facecolor','b')


figure %slopes

title ('slopes'), hold on

histogram(MeanFitResultsAI{1}(:,2),'facecolor','g'); hold on

histogram(MeanFitResultsAI{2}(:,2),'facecolor','b')


% figure %asymptote

% title ('asymptote'), hold on

% histogram(MeanFitResultsAI{1}(:,3),'facecolor','g'); hold on

% histogram(MeanFitResultsAI{2}(:,3),'facecolor','b')

% histogram(MeanFitResultsAI{3}(:,3),'facecolor','r')

% figure %fits

% title ('fits'), hold on

% histogram(MeanFitResultsAI{1}(:,4),[0:0.01:1.0],'facecolor','g'); hold on

% histogram(MeanFitResultsAI{2}(:,4),[0:0.01:1.0],'facecolor','b')

% histogram(MeanFitResultsAI{3}(:,4),[0:0.01:1.0],'facecolor','r')

%

%

figure

subplot (1,3,1)

title ('rate constants'), hold on

histogram(MeanFitResultsAI{1}(:,2),[0:0.01:0.8],'facecolor','g'); hold on

histogram(MeanFitResultsAI{2}(:,2),[0:0.01:0.8],'facecolor','b')

subplot (1,3,2)

title ('asymptote'), hold on

histogram(MeanFitResultsAI{1}(:,3),[5:0.5:20],'facecolor','g'); hold on

histogram(MeanFitResultsAI{2}(:,3),[5:0.5:20],'facecolor','b')


subplot (1,3,3)

title ('r2'), hold on

histogram(MeanFitResultsAI{1}(:,4),[0:0.01:1.0],'facecolor','g'); hold on

histogram(MeanFitResultsAI{2}(:,4),[0:0.01:1.0],'facecolor','b')


RateSig_StandardvsEnforcedTE = sum( (MeanFitResultsAI{1}(:,2) - MeanFitResultsAI{2}(:,2)) > 0)/ numel (MeanFitResultsAI{1}(:,2))


AsymSig_StandardvsEnforcedTE = sum( (MeanFitResultsAI{1}(:,3) - MeanFitResultsAI{2}(:,3)) > 0)/ numel (MeanFitResultsAI{1}(:,3))


FitsSig_StandardvsEnforcedTE = sum( (MeanFitResultsAI{1}(:,4) - MeanFitResultsAI{2}(:,4)) > 0)/ numel (MeanFitResultsAI{1}(:,3))



%% to be significant, Sig needs to be < 0.025 or > 0.9725 for alpha = 0.05 two-tailed
Sig = sum( (MeanFitResultsAI{1}(:,2) - MeanFitResultsAI{2}(:,2)) > 0)/ numel (MeanFitResultsAI{1}(:,2))

 
% function ResultForCompare
% hold-out, 50 times
clear,clc
load DDI_data.mat
tims = 50;

K_percent  =[0.85];%[ 0.15,0.25,0.50,0.75,0.85];

for k= 1: length(K_percent)
    clear AUC_S2 AUPR_S2 NC_global;
    AUC_S2= zeros(tims,1);
    AUPR_S2= zeros(tims,1);
    NC_global =  zeros(tims,1);
    AUC_Members =[];
    AUPR_Members =[];
    for r=1:tims
        disp(sprintf('-->%d -th repetition',r))
        % : new methods:
        % for fusion
%         [AUC_S2(r), AUPR_S2(r),NC_global(r),~,~,AUC_Member_cell,AUPR_Member_cell]=Predict_DDI_S2_Local(DDI,{Sd_chem+0.00000001*eye(size(Sd_chem) ),Sd_label,Sd_off_label},...
%             K_percent(k),  false, 'DS');
%         AUC_Members = [AUC_Members; cell2mat(AUC_Member_cell)];
%         AUPR_Members = [AUPR_Members; cell2mat(AUPR_Member_cell)];
% % %         [AUC_S2(r), AUPR_S2(r),NC_global(r)]=Predict_DDI_S2_Local(DDI,{Sd_label,Sd_off_label},...
% % %             K_percent(k),  false, 'DS'); % 'GLOBAL' 'Naive'
        
%% using PredictCoreByLocalModel_SingleSim in Predict_DDI_S2_Local
        % for similarity combination and single similarity
        % - average similarity
%         Sim = (Sd_chem+Sd_label+Sd_off_label)/3;
%         [AUC_S2(r), AUPR_S2(r),NC_global(r)]=Predict_DDI_S2_Local(DDI,{Sim},...
%             K_percent(k),  false);

        %-BIBM 2016 used-% 
%         [AUC_S2(r), AUPR_S2(r),NC_global(r),~,~,~,~]=...
%             Predict_DDI_S2_Local(DDI,{Sd_chem,Sd_label,Sd_off_label},...
%             K_percent(k),  false, 'lp'); close;
        
        [AUC_S2(r), AUPR_S2(r),NC_global(r),~,~,~,~]=...
            Predict_DDI_S2_Local(DDI,{Sd_chem,Sd_label,Sd_off_label},...
            K_percent(k),  false, 'DS');        



        close all
    end
    
    disp('###########################################')
    disp(K_percent(k));
    disp( [mean(AUC_S2),std(AUC_S2)])
    disp([mean(AUPR_S2),std(AUPR_S2)] )
     disp([mean(NC_global),std(NC_global)] )
     
     disp(mean(AUC_Members,1))
     disp(mean(AUPR_Members,1))
   
end
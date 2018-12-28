function [date_data, model_data, model_SWE] = RTSWE_predict(matlabdate_SN_vec, station_SWE_mat, SIERRA_WIDE_SWE_average, SIERRA_WIDE_vec, model_type)


%% Set up linear model
tic
disp('Calibrating model...');
if model_type == 0 %Fit linear model
    model_SWE = fitlm(station_SWE_mat(1:7000, :), SIERRA_WIDE_vec(1:7000));
end

if model_type == 1 %Fit regression tree
    model_SWE = fitrtree(station_SWE_mat(1:7000, :), SIERRA_WIDE_vec(1:7000), 'CrossVal','on');
end

%% Predict SN data

disp(['Running model to generate predictions...']);
tic
model_data.data_all_estimated = predict(model_SWE, station_SWE_mat);
model_data.data_all_estimated(model_data.data_all_estimated<0) = 0;
model_data.data_all_estimated(matlabdate_SN_vec>now) = NaN;

model_data.data_obs = SIERRA_WIDE_vec(1:11680);
model_data.data_pred = model_data.data_all_estimated(11681:end);
model_data.data_valid = model_data.data_all_estimated(7001:11680);
model_data.data_calib = model_data.data_all_estimated(1:7000);
toc

date_data.date_obs = matlabdate_SN_vec(1:11680);
date_data.date_pred = matlabdate_SN_vec(11681:end);
date_data.date_valid = matlabdate_SN_vec(7001:11680);
date_data.date_calib = matlabdate_SN_vec(1:7000);
% 
% % 
% % % Plot results
% % figure(1)
% % 
% % plot(matlabdate_SN_vec(1:11680), SIERRA_WIDE_vec(1:11680), '--k', 'LineWidth', 2);
% % hold on;
% % plot(matlabdate_SN_vec(1:7000), ypred(1:7000));
% % plot(matlabdate_SN_vec(7001:11680), ypred(7001:11680))
% % plot(matlabdate_SN_vec(11681:end), ypred(11681:end));
% % line([now now], [0 40],'Color', 'k');
% % grid on;
% % xlim([0 now]);
% % set(gca,'FontSize', 14);
% % xlabel('Date');
% % datetick;
% % line([now now], [0 40],'Color', 'k');
% % ylabel('Sierra Nevada Wide SWE Volume [km3]');
% % legend Real Training Validation Forecast
% % ylim([0 40]);
% 
% %% Validation metrics
% 
% corr_val = corr(ypred(7001:11680), SIERRA_WIDE_vec(7001:11680), 'rows', 'complete');
% 
% figure
% scatter(SIERRA_WIDE_vec(7001:11680), ypred(7001:11680));
% grid on;
% xlabel('Observed SWE Volume [km^3]');
% ylabel('Estimated SWE Volume [km^3]');
% set(gca, 'FontSize', 14);
% refline
% refline(1, 0);
% axis square
% box on;
% xlim([0 40]);
% ylim([0 40]);
% 
% MAE = nanmean(abs(SIERRA_WIDE_vec(7001:11680) - ypred(7001:11680)));
% RMSE = sqrt(nansum(abs(SIERRA_WIDE_vec(7001:11680) - ypred(7001:11680)).^2)./(11680-7001));
% 
% %% Generate txt data
% 
% for i = 1:length(matlabdate_SN_vec)
%     [yy, mm, dd] = datevec(matlabdate_SN_vec(i));
%     
%     
% end; 
%     
%% Generate data and send to CaDC AWS server 




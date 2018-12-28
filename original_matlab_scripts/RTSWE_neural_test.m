%% Neural network random training

for i = 28:100
    disp(['Training number ' num2str(i)]);
    ind = randperm(11680);
    ind_tra = ind(1:7000);
    ind_val = ind(7001:11680);
    
    x_tr = station_SWE_mat(ind_tra, :)';
    t_tr = SIERRA_WIDE_vec(ind_tra)';
    x_val = station_SWE_mat(ind_val, :)';
    t_val = SIERRA_WIDE_vec(ind_val)';
    x_pred = station_SWE_mat';
    
    net = fitnet(1);
    [net,tr] = train(net, x_tr, t_tr);
    model_data(i).y_val = net(x_val);
    model_data(i).y_pred = net(station_SWE_mat');
end

%% Calculate error metrics

for i = 1:100
    value_today(i) = model_data(i).y_pred(12200);
%     plot(model_data(i).y_pred, 'Color', [0.5 .5 .5]);
%     hold on;
%     plot(SIERRA_WIDE_vec(:), 'Color', 'k','LineWidth', 2);

end

figure
median_today = nanmedian(value_today);

hist(value_today,[5:0.17:7.5]);
line([median_today median_today],[0 100]);

%% Hydrologic forecasting service?
if true
 %code
   clc, clear
Stock_names = {'CAO' 'CHO' 'CHW' 'DMHL' 'ezion' 'EZRA' 'falcon' 'keppel' 'KS' 'SCI' 'SMM' 'swiber'};

    b = Stock_names{2};
    data=csvread(strcat(b,'.csv'), 1,1);
    inputSeries = tonndata(data(1:end,2:end),false,false);
    targetSeries = tonndata(data(1:end,1),false,false);
    mae = zeros(3,1);
    mape= zeros(3,1);
    rmse = zeros(3,1);
    msre = zeros(3,1);
    start = int32(5);
    last = int32(9);
    for k = 1:5 %loop for iteratively increasing the hidden layers by 5
    Performance_Matrix5 = zeros(5,4);
    s = strcat('D', int2str(start), ':', 'G', int2str(last));
    for j = 2:6 % looping through number of delays from 2 to 6
        for i = 1:3 % loop for taking average of results for each iteration
            inputDelays = 1:j;
            feedbackDelays = 1:j; 
            hiddenLayerSize = 5*k; %hiddenlayer neurons
            net = narxnet(inputDelays,feedbackDelays,hiddenLayerSize);
            net.inputs{1}.processFcns = {'removeconstantrows','mapminmax'};
            net.inputs{2}.processFcns = {'removeconstantrows','mapminmax'};
            [inputs,inputStates,layerStates,targets] = preparets(net,inputSeries,{},targetSeries);
            net.divideFcn = 'divideblock';  % Divide data randomly
            net.divideMode = 'value';  % Divide up every value
            net.divideParam.trainRatio = 60/100;
            net.divideParam.valRatio = 5/100;
            net.divideParam.testRatio = 35/100;
            net.trainFcn = 'trainlm';  % Levenberg-Marquardt
            net.performFcn = 'mse';  % Mean squared error
            net.plotFcns = {'plotperform','plottrainstate','plotresponse', ...
              'ploterrcorr', 'plotinerrcorr'};
            % Train the Network
            [net,tr] = train(net,inputs,targets,inputStates,layerStates);
            % Test the Network
            outputs = net(inputs,inputStates,layerStates);
            errors = gsubtract(targets,outputs);
            performance = perform(net,targets,outputs);
            % Recalculate Training, Validation and Test Performance
            trainTargets = gmultiply(targets,tr.trainMask);
            valTargets = gmultiply(targets,tr.valMask);
            testTargets = gmultiply(targets,tr.testMask);
            %trainPerformance = perform(net,trainTargets,outputs)
            %valPerformance = perform(net,valTargets,outputs)
            %testPerformance = perform(net,testTargets,outputs);
            % Closed Loop Network
            netc = closeloop(net);
            [xc,xic,aic,tc] = preparets(netc,inputSeries,{},targetSeries);
            netc.name = [net.name ' - Closed Loop'];
            yc = netc(xc,xic,aic);
            closedLoopPerformance = perform(netc,tc,yc);
            N_closedLoopPerformance = closedLoopPerformance/mean(cell2mat(tc));
            %Early Prediction 
            nets = removedelay(net);
            nets.name = [net.name ' - Predict One Step Ahead'];
            [xs,xis,ais,ts] = preparets(nets,inputSeries,{},targetSeries);
            ys = nets(xs,xis,ais);
            earlyPredictPerformance = perform(nets,ts,ys);
            c1 = cell2mat(ts);
            c2 = cell2mat(ys);
            b1 = c1(:,1:(end-1));
            b2 = c2(:,1:(end-1));
            N_earlyPredictPerformance = earlyPredictPerformance / mean(b1);
            % error calculation through ereperf.m external file
            MAE_ep = errperf(b1,  b2, 'mae');
            MAPE_ep = errperf(b1, b2, 'mape');
            RMSE_ep = errperf(b1, b2, 'rmse');
            MSRE_ep = errperf(b1, b2, 'msre');
        %     disp('Next day closing price was forecasted to')
        %    ys(end)
        %     figure
        %     plot([cell2mat(yc);cell2mat(tc)]')
        %     legend('Network Predictions','Expected Outputs')
        %    figure
        %    plot([cell2mat(ys);cell2mat(ts)]')
        %   legend('Network Predictions','Expected Outputs');
            mae(i) = MAE_ep;
            mape(i)= MAPE_ep;
            rmse(i) = RMSE_ep;
            msre(i) = MSRE_ep
        end
    Performance_Matrix5(j-1,1) = mean(mae);
    Performance_Matrix5(j-1,2) = mean(mape);
    Performance_Matrix5(j-1,3) = mean(rmse);
    Performance_Matrix5(j-1,4) = mean(msre);
    xlswrite(strcat(b, '_results.xlsx'), Performance_Matrix5, s);
 end
    end
    start = start+ 6;
    last = last + 6;
    end
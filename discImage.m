% IMAGE RECOGNITION ACTIVATE
% For single annealed pics
T = 196;            % id of experimental image
folder = ['C:\Users\Maxwell\Documents\Research\NNSGcombo\AlanSLSdata\T', num2str(T), 'K'];
files = dir(folder);
files = files(3:end);

id = 'BONBON';
swiss = 1;              % swiss or not! adjusts angle
N = 36;                 % sqrt number of spins

% % T 104
% scale1 = 21.5;
% scale2 = 19.5;
% theta = 0.056;
% r0 = [22 60];

% % T 120
% clf
% scale1 = 21.7;
% scale2 = 19.1;
% theta = 0.056;
% r0 = [26 71.5];

% % T 130
% clf
% scale1 = 21.9;
% scale2 = 19.1;
% theta = 0.0532;
% r0 = [28.5 71.5];

% % T 147
% clf
% scale1 = 21.9;
% scale2 = 19.1;
% theta = 0.061;
% r0 = [26 77.5];


% T 157
% scale1 = 21.9;
% scale2 = 19.5;
% theta = 0.056;
% r0 = [26 77];

% T 168
% scale1 = 21.9;
% scale2 = 19.5;
% theta = 0.056;
% r0 = [26 77];

% T 181
% scale1 = 21.9;
% scale2 = 18.5;
% theta = 0.059;
% r0 = [26 77];

% T 196
scale1 = 21.7;
scale2 = 18.5;
theta = 0.059;
r0 = [23 75];

for j = 1:length(files)
    imNow = im2double(imread([folder, '\', files(j).name]));
    [folder, '\', files(j).name]
    imNow = mean(imNow,3);
    imNow = flipud(imNow);
    
    [Nx,Ny] = size(imNow);
    imAvg = imNow;
    
    % x and y matrices
    xv = linspace(1,Nx,Nx)';
    yv = linspace(1,Ny,Ny)';
    x = zeros(Nx,Ny);
    y = zeros(Nx,Ny);
    for l = 1:Nx
        for k = 1:Ny
            x(l,k) = l;
            y(l,k) = k;
        end
    end
    
    % blur
%     a = 2;
%     g = exp(-((x).^2 + (y).^2)/a^2) + exp(-((x - Nx).^2 + (y).^2)/a^2) + exp(-((x).^2 + (y - Ny).^2)/a^2) + exp(-((x - Nx).^2 + (y - Ny).^2)/a^2);
%     
%     imAvg = ifft2(fft2(imAvg).*fft2(g));
%     imAvg = imAvg./sum(sum(imAvg));                  % Normalize
%     
    
    % Enhance contrast
    minI = min(min(imAvg));
    maxI = max(max(imAvg));
    imAvg = (imAvg - minI)/(maxI-minI);
    % color like PEEM
    imAvg = abs(2*imAvg - 1);
    
    % imAvg = imAvg.^1.2;
    % imAvg(imAvg<.15) = 0;
    
    figure(2)
    surf(x,y,imAvg-1,'edgecolor','none')
    view(0,90)
    
    title('INITIAL POSITION')
    % r0 = [125   105];
    R = [   cos(theta) -sin(theta);
        sin(theta) cos(theta) ];
    XY0 = (R*r0'*sqrt(3))';
    XY0(1) = XY0(1)/scale1;
    XY0(2) = XY0(2)/scale2;
    
    
    % Scale such that the default a = 1 and that the system is appropriately
    % rotated
    x = reshape(x,[Nx*Ny,1]);
    y = reshape(y,[Nx*Ny,1]);
    xy = (R*([x,y]'))';
    x = reshape(xy(:,1),[Nx,Ny])/scale1*sqrt(3);
    y = reshape(xy(:,2),[Nx,Ny])/scale2*sqrt(3);
    
    %%
    imAvg = imAvg.^(1);
    surf(x,y,imAvg,'edgecolor','none');
    view(0,90)
    hold on
    xx = linspace(-3,30,100);
    yy = linspace(-3,30,100);
    plot(-yy*tan(theta)*scale2/scale1,yy)
    plot((-yy*tan(theta)*scale2/scale1 + Nx*cos(theta)/scale1*sqrt(3) + Ny*sin(theta)*tan(theta)/scale2*sqrt(3)),yy)
    plot(xx,(xx*tan(theta)*scale1/scale2 + Ny*cos(theta)/scale2*sqrt(3) + Nx*sin(theta)*tan(theta)/scale1*sqrt(3)))
    plot(xx,xx*tan(theta)*scale1/scale2)
    
    % Load hopfield template to overlay
    data = load(['C:\Users\Maxwell\Documents\MATLAB\', num2str(N), 'by', num2str(N), '_spins_id_', id, '.csv']);
    xi = load(['C:\Users\Maxwell\Documents\MATLAB\', num2str(N), 'by', num2str(N), '_xi_id_', id, '.csv']);
    
    xc = data(:,1);
    yc = data(:,2);
    tht = atan2(data(:,4),data(:,3));
    Nc = length(xc);
    
    xc = xc + XY0(1);
    yc = yc + XY0(2);
    
    plot3(xc,yc,ones(Nc,1),'r.','markersize',10)
%     pause
    clf
    % Distort the lattice to better overlap with images of spins
    xLap = xc;
    yLap = yc;
    
    xcp = xc*scale1/sqrt(3);
    ycp = yc*scale2/sqrt(3);
    xycp = ((R^-1)*[xcp,ycp]')';
    xcp = xycp(:,1);
    ycp = xycp(:,2);
    
%     plot3(xc,yc,ones(Nc,1),'k.')
    % quiver3(xc - .5*data(:,4),yc - .5*data(:,5),ones(n,1),data(:,4), data(:,5),zeros(n,1),'r.')
    
%     % derivates used to gradient descend
%     xderiv = -1/12*imAvg(5:end,:) + 2/3*imAvg(4:(end-1),:) - 2/3*imAvg(2:(end-3),:) + 1/12*imAvg(1:(end-4),:);
%     yderiv = -1/12*imAvg(:,5:end) + 2/3*imAvg(:,4:(end-1)) - 2/3*imAvg(:,2:(end-3)) + 1/12*imAvg(:,1:(end-4));
%     for k = 1:Nc
%         cost = 0;
%         loop = 1;
%         xNow = xcp(k);
%         yNow = ycp(k);
%         while (cost < .9) && (loop < 1000)
%             jNow = round(xNow) - 4;
%             kNow = round(yNow) - 4;
%             grad = [xderiv(jNow,kNow);yderiv(jNow,kNow)];
%             xNow = xNow + .05*grad(1);
%             yNow = yNow + .05*grad(2);
%             cost = imAvg(jNow+4,kNow+4);
%             loop = loop + 1;
%         end
%         xLap(k) = xNow;
%         yLap(k) = yNow;
%     end
%     
%     % rotate results
%     xyLap = (R*([xLap,yLap]'))';
%     xLap = xyLap(:,1)/scale1*sqrt(3);
%     yLap = xyLap(:,2)/scale2*sqrt(3);
%     
%     plot3(xLap,yLap,ones(Nc,1),'k.')
%     pause
    

    % blur
    a = .3;
    g = exp(-((x).^2 + (y).^2)/a^2) + exp(-((x - Nx).^2 + (y).^2)/a^2) + exp(-((x).^2 + (y - Ny).^2)/a^2) + exp(-((x - Nx).^2 + (y - Ny).^2)/a^2);
    
    imNow = ifft2(fft2(imNow).*fft2(g));
    imNow = imNow./sum(sum(imNow));                  % Normalize
    

    %% Load XMCD file and extract sigma at each spin
%     imNow = im2double(imread([folder, '_XMCD_avg.png']));
    
    imNow = reshape(imNow,[Nx*Ny,1]);               % make comensurate with excised images
    imNow = imNow - mean(mean(imNow));             % separate into below/above average ranges
%     figure(2)
%     clf
%     surf(x,y,reshape(imNow,[Nx,Ny]),'edgecolor','none')
%     view(0,90)
%     hold on
    
    % Run through all spinz to find the sigmas
    sig = zeros(Nc,1);
    w = .2;
    l = .2;
    for k = 1:Nc
        %     select = ((x - xLap(k) + l/2) > (-(y - yLap(k) + w/2)*abs(tan(tht(k))))) & ((x - xLap(k) + l/2) < (-(y - yLap(k) + w/2)*abs(tan(tht(k))) + l*cos(tht(k)) + w*sin(tht(k))*abs(tan(tht(k))))) & ((y - yLap(k) + w/2) > ((x - xLap(k) + l/2)*abs(tan(tht(k))))) & ((y - yLap(k) + w/2) < ((x - xLap(k) + l/2)*abs(tan(tht(k))) + w*cos(tht(k)) + l*sin(tht(k))*abs(tan(tht(k)))));
        select = (abs(x - xLap(k)) < l/2) & (abs(y - yLap(k)) < w/2);
        %     surf(x,y,select+.1,'edgecolor','none')
        %     title(num2str(tht(k)))
        %     view(0,90)
        %     drawnow
        excise = imNow(reshape(select,[Nx*Ny,1]));
        if tht > 0
            sig(k) = sign(mean(mean(excise)));
        else
            sig(k) = -sign(mean(mean(excise)));
        end
        k
    end
    
    % Store data for each image, x y mx my
    hold on
%     quiver3(xLap-.5*sig.*cos(tht), yLap-.5*sig.*sin(tht),ones(length(xc),1),.5*sig.*cos(tht), .5*sig.*sin(tht),zeros(length(xc),1),'r');
%     drawnow
    data = [xc, yc, sig.*cos(tht), sig.*sin(tht), sig];
    csvwrite([folder,'\data_',num2str(j),'.csv'],data);
    
    %% Compare to stored patterns
    [~,npat] = size(xi);
    overlap = zeros(npat,1);
    
    for k = 1:npat
        xiNow = xi(:,k);
        overlap(k) = xiNow'*sig/Nc;
        
%         clf
%         quiver3(xLap-.5*sig.*cos(tht), yLap-.5*sig.*sin(tht),ones(length(xc),1),.5*sig.*cos(tht), .5*sig.*sin(tht),zeros(length(xc),1),'k','linewidth',4);
%         hold on
%         quiver3(xLap-.5*xiNow.*cos(tht), yLap-.5*xiNow.*sin(tht),ones(length(xc),1),.5*xiNow.*cos(tht), .5*xiNow.*sin(tht),zeros(length(xc),1),'r');
%         view(0, 90)
%         pause
    end
%     figure(4)
%     clf
%     line([-1 npat+1],std(overlap)*[1 1],'color',[.4 .4 .4])
%     hold on
%     line([-1 npat+1],-std(overlap)*[1 1],'color',[.4 .4 .4])
%     plot(overlap)
%     axis([0 npat -.3 .3])
%     box on
%     xlabel('Pattern Number','fontsize',16)
%     ylabel('Overlap','fontsize',16)
%     drawnow
end
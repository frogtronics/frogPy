function varargout = view_model_lizard(varargin)
% VIEW_MODEL_LIZARD M-file for view_model_lizard.fig
%      VIEW_MODEL_LIZARD, by itself, creates a new VIEW_MODEL_LIZARD or raises the existing
%      singleton*.
%
%      H = VIEW_MODEL_LIZARD returns the handle to a new VIEW_MODEL_LIZARD or the handle to
%      the existing singleton*.
%
%      VIEW_MODEL_LIZARD('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in VIEW_MODEL_LIZARD.M with the given input arguments.
%
%      VIEW_MODEL_LIZARD('Property','Value',...) creates a new VIEW_MODEL_LIZARD or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before view_model_lizard_OpeningFunction gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to view_model_lizard_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help view_model_lizard

% Last Modified by GUIDE v2.5 13-Jul-2010 23:36:26

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @view_model_lizard_OpeningFcn, ...
                   'gui_OutputFcn',  @view_model_lizard_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before view_model_lizard is made visible.
function view_model_lizard_OpeningFcn(hObject, eventdata, handles, varargin)

% Choose default command line output for view_model_lizard
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);




% --- Outputs from this function are returned to the command line.
function varargout = view_model_lizard_OutputFcn(hObject, eventdata, handles) 

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
axes(handles.axes1)
cla

[handles.datafilename, handles.pathname]=uigetfile({'*.txt';'*.mat'},'pick data file');

if handles.datafilename==0
    return
end

handles.datafile = fullfile(handles.pathname,handles.datafilename);
[blub1,name,handles.extd,versn] = fileparts(handles.datafile);

%set(handles.edit37_videofile,'String', handles.videofilename);
set(handles.edit1,'String', handles.datafilename);

addpath(handles.pathname)
cd(handles.pathname)

if strcmp(handles.extd,'.txt')
    fid = fopen(handles.datafile);
    C = textscan(fid,'%s');
    handles.C=C;
    if strcmp((C{1,1}{1,1}),'Field')==1
       handles.M=dlmread(handles.datafile,'\t',1,0);
       
    else
       handles.M=dlmread(handles.datafile,'\t',0,0);
    end   
   
end


guidata(hObject, handles);
mycalc(hObject, eventdata, handles)

function mycalc(hObject, eventdata, handles)



handles.time=handles.M(:,2);

handles.anklex=handles.M(:,24);
handles.ankley=handles.M(:,25);
handles.anklez=handles.M(:,26);

handles.hipX=handles.M(:,33);
handles.hipY=handles.M(:,34);
handles.hipZ=handles.M(:,35);

handles.kneeX=handles.M(:,36);
handles.kneeY=handles.M(:,37);
handles.kneeZ=handles.M(:,38);

handles.metaX=handles.M(:,18);
handles.metaY=handles.M(:,19);
handles.metaZ=handles.M(:,20);

handles.lumberX=handles.M(:,39);
handles.lumberY=handles.M(:,40);
handles.lumberZ=handles.M(:,41);

handles.stickX=handles.M(:,42);
handles.stickY=handles.M(:,43);
handles.stickZ=handles.M(:,44);

handles.caudalX=handles.M(:,27);
handles.caudalY=handles.M(:,28);
handles.caudalZ=handles.M(:,29);

handles.headX=handles.M(:,30);
handles.headY=handles.M(:,31);
handles.headZ=handles.M(:,32);

handles.toeX=handles.M(:,45);
handles.toeY=handles.M(:,46);
handles.toeZ=handles.M(:,47);

handles.count=0;

% set(handles.edit4_lumber,'String', handles.C{1,1}{39,1});
% set(handles.edit5_ankle,'String', handles.C{1,1}{26,1});

handles.totalframes=length(handles.time);

set(handles.slider1,'max',handles.totalframes, 'min',1,'Value',1);
set(handles.edit1,'String','1');
%handles.frame=1;
handles.framev=1;

guidata(hObject, handles);
mydisplay(hObject, eventdata, handles)




function mydisplay(hObject, eventdata, handles)

if handles.framev<1 
    handles.framev=1;
end
if handles.framev>handles.totalframes
    handles.frame=handles.totalframes;
end

set(handles.edit1,'String',num2str(handles.framev));

thigh=[handles.hipX(handles.framev),handles.hipY(handles.framev),handles.hipZ(handles.framev)...
    ;handles.kneeX(handles.framev),handles.kneeY(handles.framev),handles.kneeZ(handles.framev)];
shank=[handles.kneeX(handles.framev),handles.kneeY(handles.framev),handles.kneeZ(handles.framev)...
    ;handles.anklex(handles.framev),handles.ankley(handles.framev),handles.anklez(handles.framev)];
pelvis=[handles.lumberX(handles.framev),handles.lumberY(handles.framev),handles.lumberZ(handles.framev)...
    ;handles.stickX(handles.framev),handles.stickY(handles.framev),handles.stickZ(handles.framev)...
    ;handles.caudalX(handles.framev),handles.caudalY(handles.framev),handles.caudalZ(handles.framev)...
    ;handles.lumberX(handles.framev),handles.lumberY(handles.framev),handles.lumberZ(handles.framev)];
foot=[handles.anklex(handles.framev),handles.ankley(handles.framev),handles.anklez(handles.framev)...
    ;handles.metaX(handles.framev),handles.metaY(handles.framev),handles.metaZ(handles.framev)...
    ;handles.toeX(handles.framev),handles.toeY(handles.framev),handles.toeZ(handles.framev)];
hip1=[handles.lumberX(handles.framev),handles.lumberY(handles.framev),handles.lumberZ(handles.framev)...
    ;handles.hipX(handles.framev),handles.hipY(handles.framev),handles.hipZ(handles.framev)];
hip2=[handles.caudalX(handles.framev),handles.caudalY(handles.framev),handles.caudalZ(handles.framev)...
    ;handles.hipX(handles.framev),handles.hipY(handles.framev),handles.hipZ(handles.framev)];
head=[handles.lumberX(handles.framev),handles.lumberY(handles.framev),handles.lumberZ(handles.framev)...
    ;handles.headX(handles.framev),handles.headY(handles.framev),handles.headZ(handles.framev)];

xmin=0;
xmax=800;
ymin=400;
ymax=1200;
zmin=0;
zmax=800;
handles.count=handles.count+1;

axes(handles.axes1);
    plot3(thigh(:,1),thigh(:,2),thigh(:,3),'-','LineWidth',2,'MarkerEdgeColor','k','MarkerFaceColor','g','MarkerSize',10);
    hold on
    plot3(shank(:,1),shank(:,2),shank(:,3),'-','LineWidth',2,'MarkerEdgeColor','k','MarkerFaceColor','g','MarkerSize',10);
    hold on
    plot3(pelvis(:,1),pelvis(:,2),pelvis(:,3),'-','LineWidth',2,'MarkerEdgeColor','k','MarkerFaceColor','g','MarkerSize',10);
    hold on
    plot3(foot(:,1),foot(:,2),foot(:,3),'-','LineWidth',2,'MarkerEdgeColor','k','MarkerFaceColor','g','MarkerSize',10);
    hold on
    plot3(hip1(:,1),hip1(:,2),hip1(:,3),'-','LineWidth',2,'MarkerEdgeColor','k','MarkerFaceColor','g','MarkerSize',10);
    hold on
    plot3(hip2(:,1),hip2(:,2),hip2(:,3),'-','LineWidth',2,'MarkerEdgeColor','k','MarkerFaceColor','g','MarkerSize',10);
    hold on
    plot3(head(:,1),head(:,2),head(:,3),'-','LineWidth',2,'MarkerEdgeColor','k','MarkerFaceColor','g','MarkerSize',10);
    grid on
%     if handles.count==1
% %         set(handles.axes1,'CameraPositionMode','auto')
% %         set(handles.axes1,'CameraViewAngleMode','auto')
% %         set(handles.axes1,'CameraTargetMode','auto')
%         handles.camerapos = get(handles.axes1,'CameraPosition');
%         handles.cameraview = get(handles.axes1,'CameraViewAngle');
%         handles.cameratarget = get(handles.axes1,'CameraTarget');
    if handles.count>1
        set(handles.axes1,'CameraPositionMode','manual')
        set(handles.axes1,'CameraViewAngleMode','manual')
        set(handles.axes1,'CameraTargetMode','manual')
        set(handles.axes1,'CameraPosition',handles.camerapos)
        set(handles.axes1,'CameraViewAngle',handles.cameraview)
        set(handles.axes1,'CameraTarget',handles.cameratarget)
    end
    axis([xmin xmax ymin ymax zmin zmax])
%     axis vis3d 
%     plot3(handles.kneeX(handles.framev),handles.kneeY(handles.framev),handles.kneeZ(handles.framev),'--rs','LineWidth',2,'MarkerEdgeColor','k','MarkerFaceColor','b','MarkerSize',10);
%     xlabel('time');
%     ylabel('ankle');
hold off;
%         handles.camerapos = get(handles.axes1,'CameraPosition');
%         handles.cameraview = get(handles.axes1,'CameraViewAngle');
%         handles.cameratarget = get(handles.axes1,'CameraTarget');

% a = get(handles.axes1,'CameraPosition');
if handles.count==1
        handles.camerapos = get(handles.axes1,'CameraPosition');
        handles.cameraview = get(handles.axes1,'CameraViewAngle');
        handles.cameratarget = get(handles.axes1,'CameraTarget');
    set(handles.slider2,'max',10000, 'min',-10000,'Value',handles.camerapos(1));
    set(handles.slider3,'max',10000, 'min',-10000,'Value',handles.camerapos(2));
    set(handles.slider4,'max',10000, 'min',-10000,'Value',handles.camerapos(3));
end
% handles
% set(handles.axes1,'CameraPositionMode','manual')
% set(handles.axes1,'CameraTargetMode','manual')

% b=get(handles.axes1)
% % b = [200,100,0];
% % c = a-b;
% set(handles.axes1,'CameraPosition',c);


guidata(hObject, handles);





% --- Executes on slider movement.
function slider1_Callback(hObject, eventdata, handles)
handles.framev=round(get(handles.slider1,'Value'));

guidata(hObject,handles);
mydisplay(hObject, eventdata, handles)


% --- Executes during object creation, after setting all properties.
function slider1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end





function edit1_Callback(hObject, eventdata, handles)
handles.framev=round(STR2DOUBLE(get(handles.edit1,'String')));
%handles.frame=handles.trig(handles.framev);
guidata(hObject,handles);
mydisplay(hObject, eventdata, handles);


% --- Executes during object creation, after setting all properties.
function edit1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end





function edit2_Callback(hObject, eventdata, handles)
% hObject    handle to edit2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit2 as text
%        str2double(get(hObject,'String')) returns contents of edit2 as a double


% --- Executes during object creation, after setting all properties.
function edit2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end




% --- Executes on slider movement.
function slider2_Callback(hObject, eventdata, handles)
PosX=get(handles.slider2,'Value');
PosY=get(handles.slider3,'Value');
PosZ=get(handles.slider4,'Value');
handles.camerapos=[PosX,PosY,PosZ];
set(handles.axes1,'CameraPosition',handles.camerapos);

guidata(hObject,handles);



% --- Executes during object creation, after setting all properties.
function slider2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end


% --- Executes on slider movement.
function slider3_Callback(hObject, eventdata, handles)
PosX=get(handles.slider2,'Value');
PosY=get(handles.slider3,'Value');
PosZ=get(handles.slider4,'Value');
handles.camerapos=[PosX,PosY,PosZ];
set(handles.axes1,'CameraPosition',handles.camerapos);

guidata(hObject,handles);

% --- Executes during object creation, after setting all properties.
function slider3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end


% --- Executes on slider movement.
function slider4_Callback(hObject, eventdata, handles)
PosX=get(handles.slider2,'Value');
PosY=get(handles.slider3,'Value');
PosZ=get(handles.slider4,'Value');
handles.camerapos=[PosX,PosY,PosZ];
set(handles.axes1,'CameraPosition',handles.camerapos);

guidata(hObject,handles);

% --- Executes during object creation, after setting all properties.
function slider4_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end



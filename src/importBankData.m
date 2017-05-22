function [x, y, yNT] = importBankData(filename, startRow, endRow)
%IMPORTFILE Import numeric data from a text file as column vectors.
%   [AGE,JOB,MARITAL,EDUCATION,DEFAULT,BALANCE1,HOUSING,LOAN,CONTACT,DAY,MONTH,DURATION,CAMPAIGN,PDAYS,PREVIOUS,POUTCOME,Y]
%   = IMPORTFILE(FILENAME) Reads data from text file FILENAME for the
%   default selection.
%
%   [AGE,JOB,MARITAL,EDUCATION,DEFAULT,BALANCE1,HOUSING,LOAN,CONTACT,DAY,MONTH,DURATION,CAMPAIGN,PDAYS,PREVIOUS,POUTCOME,Y]
%   = IMPORTFILE(FILENAME, STARTROW, ENDROW) Reads data from rows STARTROW
%   through ENDROW of text file FILENAME.
%
% Example:
%   [age,job,marital,education,default,balance1,housing,loan,contact,day,month,duration,campaign,pdays,previous,poutcome,y] = importfile('bank.csv',2, 4522);
%
%    See also TEXTSCAN.

% Auto-generated by MATLAB on 2016/12/17 19:18:14

%% Initialize variables.
delimiter = ';';
if nargin<=2
    startRow = 2;
    endRow = inf;
end

%% Format for each line of text:
%   column1: double (%f)
%	column2: text (%q)
%   column3: text (%q)
%	column4: text (%q)
%   column5: text (%q)
%	column6: double (%f)
%   column7: text (%q)
%	column8: text (%q)
%   column9: text (%q)
%	column10: double (%f)
%   column11: text (%q)
%	column12: double (%f)
%   column13: double (%f)
%	column14: double (%f)
%   column15: double (%f)
%	column16: text (%q)
%   column17: text (%q)
% For more information, see the TEXTSCAN documentation.
formatSpec = '%f%q%q%q%q%f%q%q%q%f%q%f%f%f%f%q%q%[^\n\r]';

%% Open the text file.
fileID = fopen(filename,'r');

%% Read columns of data according to the format.
% This call is based on the structure of the file used to generate this
% code. If an error occurs for a different file, try regenerating the code
% from the Import Tool.
dataArray = textscan(fileID, formatSpec, endRow(1)-startRow(1)+1, 'Delimiter', delimiter, 'EmptyValue' ,NaN,'HeaderLines', startRow(1)-1, 'ReturnOnError', false, 'EndOfLine', '\r\n');
for block=2:length(startRow)
    frewind(fileID);
    dataArrayBlock = textscan(fileID, formatSpec, endRow(block)-startRow(block)+1, 'Delimiter', delimiter, 'EmptyValue' ,NaN,'HeaderLines', startRow(block)-1, 'ReturnOnError', false, 'EndOfLine', '\r\n');
    for col=1:length(dataArray)
        dataArray{col} = [dataArray{col};dataArrayBlock{col}];
    end
end

%% Close the text file.
fclose(fileID);

%% Post processing for unimportable data.
% No unimportable data rules were applied during the import, so no post
% processing code is included. To generate code which works for
% unimportable data, select unimportable cells in a file and regenerate the
% script.

%% Allocate imported array to column variable names
age = dataArray{:, 1};
job_original = dataArray{:, 2};
marital_original = dataArray{:, 3};
education_original = dataArray{:, 4};
default_original = dataArray{:, 5};
balance1 = dataArray{:, 6};
housing_original = dataArray{:, 7};
loan_original = dataArray{:, 8};
contact_original = dataArray{:, 9};
day = dataArray{:, 10};
month_original = dataArray{:, 11};
duration = dataArray{:, 12};
campaign = dataArray{:, 13};
pdays = dataArray{:, 14};
previous = dataArray{:, 15};
poutcome_original = dataArray{:, 16};
y_original = dataArray{:, 17};

%% Transform 'Marital' data to required format
marital = [];
for value = marital_original'
    newValue = 0;	
    if (string(value) == 'married')
        newValue = 1;
    end 
    if (string(value) == 'single')
        newValue = 2;
    end   
    if (string(value) == 'divorced')
        newValue = 3;
    end
        marital = [marital newValue];
end
%% Transposing marital vector to represent a column
marital = marital';

%% Transform 'Job' data to required format
job = [];
for value = job_original'
    newValue = 0;
    if (string(value) == 'unemployed')
        newValue = 1;
    end 
    if (string(value) == 'services')
        newValue = 2;
    end   
    if (string(value) == 'management')
        newValue = 3;
    end
    if (string(value) == 'blue-collar')
        newValue = 4;
    end
    if (string(value) == 'self-employed')
        newValue = 5;
    end
    if (string(value) == 'technician')
        newValue = 6;
    end
    if (string(value) == 'entrepreneur')
        newValue = 7;
    end
    if (string(value) == 'admin.')
        newValue = 8;
    end
    if (string(value) == 'student')
        newValue = 9;
    end
    if (string(value) == 'housemaid')
        newValue = 10;
    end
    if (string(value) == 'retired')
        newValue = 11;
    end
    if (string(value) == 'unknown')
        newValue = 12;
    end
    job = [job newValue];
end
%% Transposing job vector to represent a column
job = job';

%% Transform 'Education' data to required format
education = [];
for value = education_original'
    newValue = 0;	
    if (string(value) == 'primary')
        newValue = 1;
    end 
    if (string(value) == 'secondary')
        newValue = 2;
    end   
    if (string(value) == 'tertiary')
        newValue = 3;
    end
    if (string(value) == 'unknown')
        newValue = 4;
    end
    if (string(value) == 'education')
        newValue = 5;
    end
        education = [education newValue];
end
%% Transposing education vector to represent a column
education = education';

%% Transform 'Default' data to required format
default = [];
for value = default_original'
    newValue = 0;	
    if (string(value) == 'no')
        newValue = 0;
    end 
    if (string(value) == 'yes')
        newValue = 1;
    end
	default = [default newValue];
end
%% Transposing default vector to represent a column
default = default';

%% Transform 'Housing' data to required format
housing = [];
for value = housing_original'
    newValue = 0;	
    if (string(value) == 'no')
        newValue = 0;
    end 
    if (string(value) == 'yes')
        newValue = 1;
    end
	housing = [housing newValue];
end
%% Transposing housing vector to represent a column
housing = housing';

%% Transform 'Loan' data to required format
loan = [];
for value = loan_original'
    newValue = 0;	
    if (string(value) == 'no')
        newValue = 0;
    end 
    if (string(value) == 'yes')
        newValue = 1;
    end
	loan = [loan newValue];
end
%% Transposing loan vector to represent a column
loan = loan';

%% Transform 'Contact' data to required format
contact = [];
for value = contact_original'
    newValue = 0;	
    if (string(value) == 'unknown')
        newValue = 0;
    end 
    if (string(value) == 'cellular')
        newValue = 1;
    end
    if (string(value) == 'telephone')
        newValue = 2;
    end
	contact = [contact newValue];
end
%% Transposing contact vector to represent a column
contact = contact';

%% Transform 'Month' data to required format
month = [];
for value = month_original'
    newValue = 0;
    if (string(value) == 'jan')
        newValue = 1;
    end 
    if (string(value) == 'feb')
        newValue = 2;
    end   
    if (string(value) == 'mar')
        newValue = 3;
    end
    if (string(value) == 'apr')
        newValue = 4;
    end
    if (string(value) == 'may')
        newValue = 5;
    end
    if (string(value) == 'jun')
        newValue = 6;
    end
    if (string(value) == 'jul')
        newValue = 7;
    end
    if (string(value) == 'aug')
        newValue = 8;
    end
    if (string(value) == 'sep')
        newValue = 9;
    end
    if (string(value) == 'oct')
        newValue = 10;
    end
    if (string(value) == 'nov')
        newValue = 11;
    end
    if (string(value) == 'dec')
        newValue = 12;
    end
    month = [month newValue];
end
%% Transposing month vector to represent a column
month = month';

%% Transform 'Poutcome' data to required format
poutcome = [];
for value = poutcome_original'
    newValue = 0;	
    if (string(value) == 'unknown')
        newValue = 0;
    end 
    if (string(value) == 'failure')
        newValue = 1;
    end
    if (string(value) == 'other')
        newValue = 2;
    end
    if (string(value) == 'success')
        newValue = 3;
    end
	poutcome = [poutcome newValue];
end
%% Transposing poutcome vector to represent a column
poutcome = poutcome';

%% Transform 'Y' data to required format
y = [];
for value = y_original'
    newValue = 0;	
    if (string(value) == 'no')
        newValue = 0;
    end 
    if (string(value) == 'yes')
        newValue = 1;
    end
	y = [y newValue];
end

x = [age job marital education default balance1 housing loan contact day month duration campaign pdays previous poutcome]';

%% Generating Y values for Neural Network
yNT = [];
for value = y
    if (value == 1)
        newValue = 0;
    end 
    if (value == 0)
        newValue = 1;
    end
	yNT = [yNT newValue];
end
yNT = [y; yNT];

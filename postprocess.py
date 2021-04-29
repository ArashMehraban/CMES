import os
import sys
import pandas as pd
import numpy as np
##from collections import OrderedDict
import matplotlib.pyplot as plt

import seaborn

class AppCtx:
    pass

def parse_file_content(filename, appCtx):
    pass

def parse_filename(filename, appCtx):
    pass

def parse_log_files(folder_name, appCtx):
    #accumulate all filenames & drop the extension
    filenames=[]
    ext_sz = len(appCtx.filename_ext)
    for f in os.listdir(folder_name):
        if f[-ext_sz:] == appCtx.filename_ext:
            filenames.append(f)

    filenames_data = []
    if(appCtx.parse_filename):
        #extract data from filenames
        parse_filename = appCtx.parse_filename #function pointer
        for filename in filenames:
            filenames_data.append(parse_filename(filename,appCtx))
    
    #change directory to where log files are
    os.chdir(os.path.join(os.getcwd(),folder_name))

    #extract data from file contents
    files_data = []
    if(appCtx.parse_file_content):
        parse_file_content = appCtx.parse_file_content #function pointer
        for filename in filenames:
            files_data.append(parse_file_content(filename, appCtx)) 

    #change durectory back to where you were
    os.chdir("..")

    #return as numpy array
    return np.array(filenames_data, dtype=object), np.array(files_data, dtype=object)


def create_df(filenames_data, files_data, df_col_names, df_order, df_sort_by, df_sort_by_tuple_asc, df_drop, repeat, full_disp):
    df_vals = np.concatenate((filenames_data , files_data), axis=1)

    df_vals = df_vals[:,df_order]
    
    df = pd.DataFrame(df_vals, columns = df_col_names)
    df["#DoF"] = 3 * df["#DoF"]
    if (full_disp == False):
        pd.set_option('display.expand_frame_repr', False)
    else:
        pd.set_option("display.max_rows", None, "display.max_columns", None, 'display.width', None)

    df = df.sort_values(df_sort_by, ascending = df_sort_by_tuple_asc)

    df_tmp = df.to_numpy()
    r,c = df_tmp.shape

    df_np_vals = np.zeros((int(r/repeat), int(c-len(df_drop))))
    k=0
    for i in range(0,r,repeat):
        for j in range(repeat):
            df_np_vals[k] += np.asarray((df_tmp[i+j,0:-1]), dtype=np.float64)/repeat
        k=k+1

    for item in df_drop:
        del df[item]

    #create a final dataframe to return
    dff = pd.DataFrame(df_np_vals, columns = df. columns)
    dff["L2 Error"] = 1

    all_strain = dff['Strain Energy'].to_numpy()
    L2_err = abs(all_strain - all_strain[-1])/abs(all_strain[-1])

    dff["L2 Error"] = L2_err

    dff["#Refine"] = dff["#Refine"].astype(int)
    dff["deg"] = dff["deg"].astype(int)
    dff["#CG"] = dff["#CG"].astype(int)
    dff["#DoF"] = dff["#DoF"].astype(int)
    dff["np"] = dff["np"].astype(int)
    #dff['L2 Error'] = dff['L2 Error'].apply(lambda x: '%.3e' % x)
    return dff

def plot_cost_err_seaborn(df, filename=None,nu=None,Ylim=None):
    #df.rename(columns={'Solve Time(s)': 'Solve Time (s)'}, inplace=True)
    df['Cost'] = df['Solve Time(s)'] * df['np']
    df.drop(df.tail(1).index,inplace=True)
    #print(df.tail())
    grid = seaborn.relplot(
        data=df,
        x='Cost',
        y='L2 Error',
        hue='deg',
        size='np',
        sizes=(30, 500),
        alpha=0.7,
        palette='colorblind',
    )
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(Ylim)
    plt.ylabel(r'Relative $L^2$ Error')
    plt.title(r'$\nu$ = {}'.format(nu))
    grid.tight_layout()
    if filename:
        plt.savefig(filename)
    plt.show()

def plot_time_err_seaborn(df, filename=None, nu= None, Ylim=None):
    df.drop(df.tail(1).index,inplace=True)
    #print(df.tail())
    grid = seaborn.relplot(
        data=df,
        x='Solve Time(s)',
        y='L2 Error',
        hue='deg',
        size='np',
        sizes=(30, 500),
        alpha=0.7,
        palette='colorblind',
    )
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(Ylim)
    plt.ylabel(r'Relative $L^2$ Error')
    plt.title(r'$\nu$ = {}'.format(nu))
    grid.tight_layout()
    if filename:
        plt.savefig(filename)
    plt.show()
        
##def draw_time_table(df):
##    #filters:
##    #p
##    
##    p1 = df['deg']==1
##    p2 = df['deg']==2
##    p3 = df['deg']==3
##    p4 = df['deg']==4
##    ps = [p1,p2,p3,p4]
##    #alg 
##    alg1 = df['Alg']==1
##    alg2 = df['Alg']==2
##    alg3 = df['Alg']==3
##    alg4 = df['Alg']==4
##    algs = [alg1,alg2,alg3,alg4]
##    time = np.zeros((4,4))
##
##    for i in range(len(ps)):
##        for j in range(len(algs)):
##             time[i][j] = np.round(df.where((ps[i] & algs[j]))['Solve Time(s)'].dropna(), decimals = 2)
##    print(time)


def draw_paper_data_tube(df,deg):
    mdf = df.drop(['#CG','MDoFs/Sec','Petsc Time(s)', 'Solve Time(s)','Total Time(s)','np'], axis=1)
    tmp = mdf.groupby(['deg','#Refine'],as_index = False).first()
    tmp_p = tmp.copy()
    tmp_p['L2 Error'] = tmp_p['L2 Error'].apply(lambda x: '%.3e' % x)
    print(tmp_p.to_latex())
    hd = [[0.7411 ,   0.4941  ,  0.3705  ,  0.2964],\
          [0.7411 ,   0.4941   , 0.3705  ,  0.2964],\
          [2.9620  ,  1.9756  ,  1.4819 ,   1.1856  ,  0.9880], \
          [2.9620  ,  1.9756   , 1.4819 ,   1.1856]]

    tmp = tmp[:-1] 
    if(deg == 4):
        err1 = tmp.where(tmp['deg']==1)['L2 Error'].dropna()
    err2 = tmp.where(tmp['deg']==2)['L2 Error'].dropna()
    err3 = tmp.where(tmp['deg']==3)['L2 Error'].dropna()
    err4 = tmp.where(tmp['deg']==4)['L2 Error'].dropna()

    if(deg == 4):
        err = [err1,err2,err3,err4]
    else:
        err = [err2,err3,err4]
        hd = hd[1:]

    convergence_rate = []   
    for i in range(len(hd)):
        s,bb = lin_reg_fit(np.log10(hd[i]), np.log10(err[i]))
        convergence_rate.append(round(s, 2))
    print(convergence_rate)
   

def sort_by(df, sortby):
    return df.sort_values([sortby], ascending = (True))

def draw_paper_data_beam(df):
    mdf = df.drop(['#CG','MDoFs/Sec','Petsc Time(s)', 'Solve Time(s)','Total Time(s)','np'], axis=1)
    #mdf['Solve Time(s)'] = np.round(mdf['Solve Time(s)'], decimals=2)
    mdf['Strain Energy'] = mdf['Strain Energy'].apply(lambda x: '%.6e' % x)
    mdf['L2 Error'] = mdf['L2 Error'].apply(lambda x: '%.3e' % x)
    print(mdf.to_latex())
    return mdf

def process_log_files_linE_beam(folder_name, filename_ext, keep_idx, logfile_keywords,repeat,full_disp):
    
    appCtx=AppCtx()
    #filename attributes for appCtx
    appCtx.filename_ext = filename_ext
    appCtx.keep_idx = keep_idx
    appCtx.parse_filename = parse_filename_linE #function pointer
    
    #file content attributes for appCtx
    appCtx.parse_file_content = parse_file_content_linE_beam #function pointer
    appCtx.logfile_keywords = logfile_keywords
    appCtx.repeat = repeat

    #parse files and filenames
    filenames_data , files_data = parse_log_files(folder_name, appCtx)

    #data frame info:
    df_col_names = ['#Refine', 'deg', '#DoF', '#CG','Solve Time(s)','MDoFs/Sec', 'Strain Energy','Petsc Time(s)', 'Total Time(s)','np','run']
    df_order = [0,1,4,5,6,7,8,9,10,2,3]
    df_sort_by = ['deg', '#Refine', 'np', 'run']
    df_sort_by_tuple_asc = (True, True,True,True)  
    df_drop = ['run']
    repeat = 3
    #create a dataframe
    df = create_df(filenames_data, files_data, df_col_names, df_order, df_sort_by, df_sort_by_tuple_asc, df_drop, repeat, full_disp)
    return df

def process_log_files_linE_tube(folder_name, filename_ext, keep_idx, logfile_keywords,repeat,full_disp):

    appCtx=AppCtx()
    #filename attributes for appCtx
    appCtx.filename_ext = filename_ext
    appCtx.keep_idx = keep_idx
    appCtx.parse_filename = parse_filename_linE #function pointer
    
    #file content attributes for appCtx
    appCtx.parse_file_content = parse_file_content_linE_tube #function pointer parse_file_content_NH_noether
    appCtx.logfile_keywords = logfile_keywords
    appCtx.repeat = repeat

    #parse files and filenames
    filenames_data , files_data = parse_log_files(folder_name, appCtx)

    #data frame info:
    df_col_names = ['#Refine', 'deg', '#DoF', '#CG','Solve Time(s)','MDoFs/Sec', 'Strain Energy','Petsc Time(s)', 'Total Time(s)','np','run']
    df_order = [0,1,3,4,5,6,7,9,10,8,2]
    df_sort_by = ['deg', '#Refine', 'np', 'run']
    df_sort_by_tuple_asc = (True, True,True,True)  
    df_drop = ['run']
    repeat = 3
    #create a dataframe
    df = create_df(filenames_data, files_data, df_col_names, df_order, df_sort_by, df_sort_by_tuple_asc, df_drop, repeat, full_disp)
    return df


def parse_file_content_linE_tube(filename, appCtx):
    grep = appCtx.logfile_keywords
    file_data = []
    fd = open(filename, 'r')
    lines = fd.readlines()
    for line in lines:
        ll = line.strip().split()
        if grep[0] in line:
            file_data.append(int(ll[-1]))  #node
        elif grep[1] in line:
            file_data.append(int(ll[-1]))  #ksp
        elif grep[2] in line: 
            file_data.append(float(ll[-3])) #snes time
        elif grep[3] in line:
            file_data.append(float(ll[-3])) #Dof/Sec
        elif grep[4] in line:
            file_data.append(float(ll[-1])) #"Strain
        elif grep[5] in line:
            file_data.append(int(ll[7])) #cpu 
        elif grep[6] in line:
            file_data.append(float(ll[2]))  #petsc total time  
        elif grep[7] in line:
            file_data.append(float(ll[-1])) #script time                        
    if len(file_data) < len(grep):
        print('Not enough data recored for:')
        print(filename)
    fd.close()
    return file_data


def parse_file_content_linE_beam(filename, appCtx):
    grep = appCtx.logfile_keywords
    file_data = []
    fd = open(filename, 'r')
    lines = fd.readlines()
    for line in lines:
        ll = line.strip().split()
        if grep[0] in line:
            file_data.append(int(ll[-1]))  #node
        elif grep[1] in line:
            file_data.append(int(ll[-1]))  #ksp
        elif grep[2] in line: 
            file_data.append(float(ll[-3])) #snes time
        elif grep[3] in line:
            file_data.append(float(ll[-3])) #Dof/Sec
        elif grep[4] in line:
            file_data.append(float(ll[-1])) #"Strain
        elif grep[5] in line:
            file_data.append(float(ll[2]))  #petsc total time  
        elif grep[6] in line:
            file_data.append(float(ll[-1])) #script time                        
    if len(file_data) < len(grep):
        print('Not enough data recored for:')
        print(filename)
    fd.close()
    return file_data


def parse_filename_linE(filename,appCtx):
    ext_sz = len(appCtx.filename_ext)
    f = filename[:-ext_sz].split('_')
    data = []    
    for i in range(len(f)):
        if i in appCtx.keep_idx:
            if f[i].isdigit() or f[i].replace('.', '', 1).isdigit():
                data.append(digitize(f[i]))
    return data


def digitize(item):
    if '.' in item:
        item.replace('.', '', 1).isdigit()
        return float(item)
    elif item.isdigit():
        return int(item)
    else:
        return

def lin_reg_fit(x,y):

    if x.shape != y.shape:
        print('input size mismatch')
    else:
        n = x.size
    xy = x * y
    x_sq = x**2
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x*y)
    sum_x_sq = np.sum(x_sq)
    #slope
    m = (n * sum_xy - sum_x * sum_y) /(n * sum_x_sq - sum_x**2)
    #b
    b = (sum_y - m * sum_x) / n
    return m, b

def compute_conv_slope(df,h):
    convergence_rate = []
    df = df[:-1]
    for i in range(1,len(h)+1):
        err = df.where(df['deg']==i)['L2 Error'].dropna()
        s,bb = lin_reg_fit(np.log10(h), np.log10(err))
        convergence_rate.append(round(s, 2))
    return convergence_rate
        
     

if __name__ == "__main__":

    #log files' extension
    filename_ext = '.log'
    #number of repeats per simulation
    repeat = 3

                                              #Compressible Tube 
    #---------------------------------------------------------------------------------------------------
    folder_name = 'log_files_tube_comp'
    #indecies to keep from filename
    #idx:    0   1    2  3  4  5  6  7  8 
    #     Tube8_20int_1_deg_3_cpu_1_run_1.log
    keep_idx = [2,4,8]  
    logfile_keywords = ['Global nodes', 'Total KSP Iterations', 'SNES Solve Time', 'DoFs/Sec in SNES', \
                        'Strain Energy', '.edu with','Time (sec):','script']
                                        #line containing .edu with has number of processors
    full_disp = True
    df = process_log_files_linE_tube(folder_name, filename_ext, keep_idx, logfile_keywords,repeat,full_disp)
    print(df)
##  draw_paper_data(df)
    nu = 0.3
    ylim = [0.00001, 0.1]
##    plot_cost_err_seaborn(df, 'error-cost-tube-comp.png',nu,ylim)
##    plot_time_err_seaborn(df, 'error-time-tube-comp.png',nu,ylim)
    draw_paper_data_tube(df,4)
    #---------------------------------------------------------------------------------------------------
##    
##
                                            #Incompressible Tube
    #---------------------------------------------------------------------------------------------------
    folder_name = 'log_files_tube_incomp'
    #indecies to keep from filename
    #idx:    0   1    2  3  4  5   6    7     8  9
    #     Tube8_20int_1_deg_3_cpu_384_incomp_run_2.log
    logfile_keywords = ['Global nodes','Total KSP Iterations', 'SNES Solve Time', \
                        'DoFs/Sec in SNES', 'Strain Energy', './elasticity', 'Time (sec):', 'script']
    keep_idx = [2,4,9]
    full_disp = True
    df = process_log_files_linE_tube(folder_name, filename_ext, keep_idx, logfile_keywords,repeat,full_disp)
    #print(df)
    nu = 0.499999
    ylim = [.6, 1]
    #plot_cost_err_seaborn(df, 'error-cost-tube-incomp.png',nu,ylim)
    #plot_time_err_seaborn(df, 'error-time-tube-incomp.png',nu,ylim)
    draw_paper_data_tube(df,3)
    #df.to_csv (r'compressible.csv', index = False, header=True)
    #---------------------------------------------------------------------------------------------------


##                                                   #Beam
##    #---------------------------------------------------------------------------------------------------
##    folder_name = 'log_files_beam'
##    filename_ext = '.log'
##    #idx: 0   1   2  3  4  5  6   7  8 
##    #     23_Beam_3_deg_2_cpu_64_run_3.log
##    keep_idx = [2,4,6,8]
##
##    logfile_keywords = ['Global nodes','Total KSP Iterations', 'SNES Solve Time', \
##                        'DoFs/Sec in SNES', 'Strain Energy', 'Time (sec):', 'script']
##    full_disp = True
##    df = process_log_files_linE_beam(folder_name, filename_ext, keep_idx, logfile_keywords,repeat,full_disp)
##    print(df)
##    draw_paper_data_beam(df)
##    h = [0.1428, 0.0714, 0.0476, 0.0357]
##    print("slopes for beam with poly order 1-4 ")
##    cs = compute_conv_slope(df,h)
##    print(cs)
    
    #---------------------------------------------------------------------------------------------------

# Beam h sizes: 0.1428    0.0714    0.0476    0.0357    0.0089
    
    


    

    

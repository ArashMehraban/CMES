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


def create_df(filenames_data, files_data, df_col_names, df_order, df_sort_by, df_sort_by_tuple_asc, df_drop, repeat, dof, full_disp):
    df_vals = np.concatenate((filenames_data , files_data), axis=1)

    df_vals = df_vals[:,df_order]
    
    df = pd.DataFrame(df_vals, columns = df_col_names)
    df["#DoF"] = dof * df["#DoF"]
    if (full_disp == False):
        pd.set_option('display.float_format','{:.5f}'.format)
    else:
        pd.set_option("display.max_rows", None, "display.max_columns", None, 'display.width', None)


    df = df.sort_values(df_sort_by, ascending = df_sort_by_tuple_asc).reset_index(drop=True)                               

    df_tmp = df.to_numpy()
    r,c = df_tmp.shape

    if(r%repeat != 0):
        print('Fatal Error. The number of rows in the dataframe: {}, is not divisiable by {}. Here is your dataframe:'.format(r,repeat), file=sys.stderr)
        print(df)
        sys.exit()

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
    dff.index = dff.index + 1 # <-- make rows to show up from 1 to n instead of 0 to n-1 

    dff["#Refine"] = dff["#Refine"].astype(int)
    dff["deg"] = dff["deg"].astype(int)
    dff["#CG"] = dff["#CG"].astype(int)
    dff["#DoF"] = dff["#DoF"].astype(int)
    dff["np"] = dff["np"].astype(int)
    dff['Solve Time(s)'] = dff['Solve Time(s)'].round(2)
    #dff['Total Time(s)'] = dff['Total Time(s)'].round(2)
    dff['Petsc Time(s)'] = dff['Petsc Time(s)'].round(2)
    #dff['Strain Energy'] = dff['Strain Energy'].apply(lambda x: '%.2f' % x) <-- This change the type to string (Do NOT use!)
    return dff

def plot_cost_err_seaborn(df, filename=None,nu=None,Ylim=None):
    df['Cost'] = df['Solve Time(s)'] * df['np']
    dff = df.copy(deep=True)
    dff.drop(df.tail(1).index,inplace=True)
    grid = seaborn.relplot(
        data=dff,
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
    del df['Cost']

def plot_time_err_seaborn(df, filename=None, nu= None, Ylim=None):
    dff = df.copy(deep=True)
    dff.drop(df.tail(1).index,inplace=True)
    grid = seaborn.relplot(
        data=dff,
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


def draw_paper_data_tube(df): #,deg):
    mdf = df.drop(['#CG','MDoFs/Sec','Petsc Time(s)', 'Solve Time(s)','np'], axis=1)
    mdf = df.drop(['#CG','MDoFs/Sec','Petsc Time(s)', 'Solve Time(s)','np'], axis=1)
    tmp = mdf.groupby(['deg','#Refine'],as_index = False).first()
    tmp['Strain Energy'] = tmp['Strain Energy'].apply(lambda x: '%.4e' % x)
    #tmp['L2 Error'] = tmp['L2 Error'].apply(lambda x: '%.3e' % x)    
    print(tmp.to_latex(index=False))

##    ##NOTE: Hard Coded below (hd values come from MATLAB code):
##    print("WARNING: Hard Coded for (element size h):")
##    print("This function will break if you change the data!!")
##    print("FIX later")
##    #refine 1        2          3         4        7          11      15         19
##    #    0.0030    0.0020    0.0015    0.0012    0.0007    0.0005    0.0004    0.0003
##    hd = [[0.0030 ,   0.0020 ,   0.0015  ,  0.0012],\
##          [0.0030 ,   0.0020 ,   0.0015  ,  0.0012],\
##          [0.0007  ,  0.0005  ,  0.0004 ,   0.0003], \
##          [0.0007  ,  0.0005  ,  0.0004 ,   0.0003]]
##
##    tmp['L2 Error'] = tmp['L2 Error'].astype(float)
##    if(deg == 4):
##        err1 = tmp.where(tmp['deg']==1)['L2 Error'].dropna()
##    err2 = tmp.where(tmp['deg']==2)['L2 Error'].dropna()
##    err3 = tmp.where(tmp['deg']==3)['L2 Error'].dropna()
##    err4 = tmp.where(tmp['deg']==4)['L2 Error'].dropna()
##
##    if(deg == 4):
##        err = [err1,err2,err3,err4[:-1]]
##    else:
##        err = [err2,err3,err4[:-1]]
##        hd = hd[1:]
##    
##    convergence_rate = []   
##    for i in range(len(hd)):
##        s,bb = lin_reg_fit(np.log10(hd[i]), np.log10(err[i]))
##        convergence_rate.append(round(s, 2))
##    print("Convergence rates:")
##    print(convergence_rate)
   

def sort_by(df, sortby):
    return df.sort_values([sortby], ascending = (True))

def compute_error(df,refine,p):
    #for p-refinement
    pdf = df.copy(deep=True)
    
    #h-refinement error
    #hdf = df.drop(drop_col_names, axis=1)
    numRows = df.shape[0]
    err_h = np.zeros(numRows)
    for i in range(0,len(p)): #for each poly order do:
        strain = df.where(df['deg']==p[i])['Strain Energy'].dropna()
        strain_sz = strain.size
        finest = strain.iloc[strain_sz-1]
        err_h[i*strain_sz:(i+1)*strain_sz] = abs(strain - finest)/abs(finest)
    df['L2 Error(h)'] = err_h
    

    #p-refinement error
    pdf = pdf.sort_values(['#Refine', 'deg', '#DoF', 'Strain Energy'], ascending = (True, True,True, True)).reset_index(drop=True)
    pdf.index = pdf.index + 1
    err_p = np.zeros(numRows)
    se_beg = 0
    for i in range(0,len(refine)): #for each refinement do:
        strain = pdf.where(pdf['#Refine']==refine[i])['Strain Energy'].dropna()
        strain_sz = strain.size
        finest = strain.iloc[strain_sz-1]
        se_end = se_beg + strain_sz
        err_p[se_beg:se_end] = abs(strain - finest)/abs(finest)
        se_beg = se_end
    pdf['L2 Error(p)'] = err_p

    #hp-refinement error
    err_p_based_on_larget_hp = np.zeros(numRows)
    strain_all = pdf['Strain Energy'].dropna().astype(np.float)
    last = strain_all.iloc[numRows-1]
    err_p_based_on_larget_hp = abs(strain_all - last )/abs(last)
    pdf['L2 Error(hp)'] = err_p_based_on_larget_hp

    return df, pdf
    

def draw_paper_data_beam(df, pdf, drop_col_names):
    hdf = df.drop(drop_col_names, axis=1)
    hdf['L2 Error(h)'] = hdf['L2 Error(h)'].apply(lambda x: '%.4e' % x)
    hdf['Strain Energy'] = hdf['Strain Energy'].apply(lambda x: '%.6e' % x)
    print('L2 ERROR based on h-refinement:\n')
    print(hdf.to_latex(index=False))

    pdf = pdf.drop(drop_col_names, axis=1)
    pdf['L2 Error(p)'] = pdf['L2 Error(p)'].apply(lambda x: '%.4e' % x)
    pdf['Strain Energy'] = pdf['Strain Energy'].apply(lambda x: '%.6e' % x)
    print('L2 ERROR based on p and hp-refinement:\n')
    pdf['L2 Error(hp)'] = pdf['L2 Error(hp)'].apply(lambda x: '%.4e' % x)
    print(pdf.to_latex(index=False))
    
    
    

def process_log_files_linE_beam(folder_name, filename_ext, keep_idx, logfile_keywords,repeat,dof, full_disp):
    
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
    df_col_names = ['#Refine', 'deg', '#DoF', '#CG','Solve Time(s)','MDoFs/Sec', 'Strain Energy','Petsc Time(s)','np','run']
    df_order = [0,1,4,5,6,7,8,9,2,3]
    df_sort_by = ['deg', '#Refine', 'np', 'run']
    df_sort_by_tuple_asc = (True, True,True,True)  
    df_cols_drop = ['run']
    repeat = 3
    #create a dataframe
    df = create_df(filenames_data, files_data, df_col_names, df_order, df_sort_by, df_sort_by_tuple_asc, df_cols_drop, repeat, dof, full_disp)
    return df

def process_log_files_linE_tube(folder_name, filename_ext, keep_idx, logfile_keywords,repeat,dof,full_disp):

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
    df_col_names = ['#Refine', 'deg', '#DoF', '#CG','Solve Time(s)','MDoFs/Sec', 'Strain Energy','Petsc Time(s)','np','run']
    df_order = [0,1,3,4,5,6,7,9,8,2]
    df_sort_by = ['deg', '#Refine', 'np', 'run']
    df_sort_by_tuple_asc = (True, True,True,True)  
    dof_cols_drop = ['run']
    repeat = 3
    #create a dataframe
    df = create_df(filenames_data, files_data, df_col_names, df_order, df_sort_by, df_sort_by_tuple_asc, dof_cols_drop, repeat, dof, full_disp)
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
##        elif grep[7] in line:
##            file_data.append(float(ll[-1])) #script time                        
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
        #elif grep[6] in line:
         #   file_data.append(float(ll[-1])) #script time                        
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

    print('-------------------------------Compressible-----------------------------------------------')

                                              #Compressible Tube 
    #---------------------------------------------------------------------------------------------------
    folder_name = 'log_files_tube_comp'
    #indecies to keep from filename
    #idx:    0   1    2  3  4  5  6  7  8 
    #     Tube8_20int_1_deg_3_cpu_1_run_1.log
    keep_idx = [2,4,8]  
    logfile_keywords = ['Global nodes', 'Total KSP Iterations', 'SNES Solve Time', 'DoFs/Sec in SNES', \
                        'Strain Energy', '.edu with','Time (sec):']
                                        #line containing .edu with has number of processors
    full_disp = True
    dof = 3
    df = process_log_files_linE_tube(folder_name, filename_ext, keep_idx, logfile_keywords,repeat,dof, full_disp)
    refine = [1,2,3,4,5]
    p = [1,2,3,4]
    df, pdf = compute_error(df,refine,p)
    print(df)
    h = [0.0030, 0.0020, 0.0015, 0.0012 ,0.0010]
    nu = 0.3
    ylim = [0.00001, 0.1]
    #plot_cost_err_seaborn(df, 'error-cost-tube-comp.png',nu,ylim)
    #plot_time_err_seaborn(df, 'error-time-tube-comp.png',nu,ylim)
    #draw_paper_data_tube(df,4) #<---- 4 mean use poly orders 1,2,3 and 4
    #---------------------------------------------------------------------------------------------------
    
    print('-------------------------------Incompressible-----------------------------------------------')
                                            #Incompressible Tube
    #---------------------------------------------------------------------------------------------------
    folder_name = 'log_files_tube_incomp'
    #indecies to keep from filename
    #idx:    0   1    2  3  4  5   6    7     8  9
    #     Tube8_20int_1_deg_3_cpu_384_incomp_run_2.log
    logfile_keywords = ['Global nodes','Total KSP Iterations', 'SNES Solve Time', \
                        'DoFs/Sec in SNES', 'Strain Energy', './elasticity', 'Time (sec):']
    keep_idx = [2,4,9]
    full_disp = True
    dof = 3
    df = process_log_files_linE_tube(folder_name, filename_ext, keep_idx, logfile_keywords,repeat, dof, full_disp)
    refine = [1,2,3,4,5]
    p = [2,3,4]
    df, pdf = compute_error(df,refine,p)
    print(df)
    h = [0.0030, 0.0020, 0.0015, 0.0012 ,0.0010]
    nu = 0.499999
    ylim = [0.00001, 0.1] #[.6, 1]
    #plot_cost_err_seaborn(df, 'error-cost-tube-incomp.png',nu,ylim)
    #plot_time_err_seaborn(df, 'error-time-tube-incomp.png',nu,ylim)
    #draw_paper_data_tube(df,3)  #<---- 3 mean use poly orders 2,3 and 4
    #---------------------------------------------------------------------------------------------------

    print('-------------------------------Beam-----------------------------------------------')

                                                   #Beam
    #---------------------------------------------------------------------------------------------------
    folder_name = 'log_files_beam'
    filename_ext = '.log'
    #idx: 0   1   2  3  4  5  6   7  8 
    #     23_Beam_3_deg_2_cpu_64_run_3.log
    keep_idx = [2,4,6,8]

    logfile_keywords = ['Global nodes','Total KSP Iterations', 'SNES Solve Time', \
                        'DoFs/Sec in SNES', 'Strain Energy', 'Time (sec):']
    full_disp = True
    dof = 3
    df = process_log_files_linE_beam(folder_name, filename_ext, keep_idx, logfile_keywords, repeat, dof, full_disp)
    refine = [0,1,2,3,4]
    p = [1,2,3,4]
    df, pdf = compute_error(df,refine,p)
    drop_col_names = ['#CG','MDoFs/Sec','Petsc Time(s)', 'Solve Time(s)','np']
    draw_paper_data_beam(df,pdf, drop_col_names)
    h = [0.1428, 0.0714, 0.0476 ,0.0357, 0.0286]
##    h = [0.1428, 0.0714, 0.0476, 0.0357]
##    print("slopes for beam with poly order 1-4 ")
##    cs = compute_conv_slope(df,h)
##    print(cs)
    
    #---------------------------------------------------------------------------------------------------

# Beam h sizes(Beam8 0-4 refinements): 0.1428    0.0714    0.0476    0.0357    0.0286 

# Tube h sizes (Tube8_20_{1-5} refinements): 0.0030    0.0020    0.0015    0.0012    0.0010    
    


    

    

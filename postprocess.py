import os
import sys
import pandas as pd
import numpy as np
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
    dff['Petsc Time(s)'] = dff['Petsc Time(s)'].round(2)
    #dff['Strain Energy'] = dff['Strain Energy'].apply(lambda x: '%.2f' % x) <-- This change the type to string (Do NOT use!)
    return dff

def plot_cost_err_seaborn(df, x=None, y=None, filename=None,nu=None,Ylim=None):
    dff = df[(df[[y]] != 0).all(axis=1)]
    df['Cost'] = df['Solve Time(s)'] * df['np']
    dff = df.copy(deep=True)
    dff.drop(df.tail(1).index,inplace=True)
    grid = seaborn.relplot(
        data=dff,
        x=x,
        y=y,
        hue='deg',
        size='np',
        sizes=(30, 500),
        alpha=0.7,
        palette='colorblind',
    )
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(Ylim)
    plt.ylabel(r'Relative $L^2$ Error (' + y[-2:-1] + ')')
    plt.title(r'$\nu$ = {}'.format(nu))
    grid.tight_layout()
    if filename:
        plt.savefig(filename)
    plt.show()
    del df['Cost']

def plot_time_err_seaborn(df, x=None, y=None, filename=None, nu= None, Ylim=None):
    dff = df[(df[[y]] != 0).all(axis=1)]
    grid = seaborn.relplot(
        data=dff,
        x=x,
        y=y,
        hue='deg',
        size='np',
        sizes=(30, 500),
        alpha=0.7,
        palette='colorblind',
    )
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(Ylim)
    plt.ylabel(r'Relative $L^2$ Error (' + y[-2:-1] + ')')
    plt.title(r'$\nu$ = {}'.format(nu))
    grid.tight_layout()
    if filename:
        plt.savefig(filename)
    plt.show()
        

def plot_beam_p_conv(pdf, refine ,ps):
    
    plt_marker = [ '*','o', '^', 'p','+']
    plt_linestyle = ['--g','-.r', ':b', '--k','-.m']
    fig, ax = plt.subplots()
    for i in range(len(refine)):
        err = np.array(pdf.where(pdf['#Refine']==refine[i])['L2 Error(p)'].dropna() ,dtype=float)[:-1]
        sz_err = err.shape[0]
        ax.semilogy(ps[0:sz_err],err,plt_linestyle[i], marker=plt_marker[i], label='Refinement {}'.format(i))
    plt.grid()
    ax.set_xticks((1,2,3))
    plt.title('Error vs. p (semilogy)')
    plt.legend(ncol = 2, loc="upper right", shadow=True)
    plt.xlabel('polynomial order')
    plt.ylabel(r'$L^2$ Error', rotation=90)
    plt.show()

def plot_beam_h_conv(df, ps ,hs, filename=None):
    convergence_rate =[]
    plt_marker = [ '*','o', '^', 'p']
    plt_linestyle = ['--g','-.r', ':b', '--k']
    fig, ax = plt.subplots()
    for i in range(len(ps)):
        err = np.array(df.where(df['deg']==ps[i])['L2 Error(h)'].dropna() ,dtype=float)[:-1]
        sz_err = err.shape[0]
        ax.loglog(np.array(h[0:sz_err]), err ,plt_linestyle[i], marker=plt_marker[i], label='p{}'.format(i+1))
        s,bb = lin_reg_fit(np.log10(h[0:sz_err]), np.log10(err))
        convergence_rate.append(round(s, 2))
    
    plt.title('Error vs. h (loglog)')
    plt.legend(ncol = 2, loc="upper left", shadow=True)
    plt.xlabel('h')
    plt.ylabel(r'$L^2$ Error', rotation=90)
    plt.grid()
    plt.show()
    if filename:
        plt.savefig(filename)
    print(convergence_rate)

def compute_error(df,refine,p,roundFlag):
    #for p-refinement
    pdf = df.copy(deep=True)
    
    #h-refinement error
    #hdf = df.drop(drop_col_names, axis=1)
    numRows = df.shape[0]
    err_h = np.zeros(numRows)
    se_beg = 0
    for i in range(len(p)): #for each poly order do:
        strain = df.where(df['deg']==p[i])['Strain Energy'].dropna()
        strain_sz = strain.size
        finest = strain.iloc[strain_sz-1]
        se_end = se_beg + strain_sz
        err_h[se_beg:se_end] = abs(strain - finest)/abs(finest)
        se_beg = se_end
    df['L2 Error(h)'] = err_h   

    #p-refinement error
    pdf = pdf.sort_values(['#Refine','deg', '#DoF', 'Strain Energy'], ascending = (True, True,True, True)).reset_index(drop=True)
    pdf.index = pdf.index + 1
    err_p = np.zeros(numRows)
    se_beg = 0

    pd.options.display.float_format = "{:,.15f}".format
    for i in range(len(refine)): #for each refinement do:
        strain = pdf.where(pdf['#Refine']==refine[i])['Strain Energy'].dropna()
        strain_sz = strain.size
        finest = strain.iloc[strain_sz-1]
        se_end = se_beg + strain_sz
        if(roundFlag):
            err_p[se_beg:se_end] = abs(round(strain,7) - round(finest,7))/abs(round(finest,7))
        else:
            err_p[se_beg:se_end] = abs(strain - finest)/abs(finest)
        se_beg = se_end
    pdf['L2 Error(p)'] = err_p

##    #hp-refinement error
##    err_p_based_on_larget_hp = np.zeros(numRows)
##    strain_all = pdf['Strain Energy'].dropna().astype(np.float)
##    last = strain_all.iloc[numRows-1]
##    err_p_based_on_larget_hp = abs(strain_all - last )/abs(last)
##    pdf['L2 Error(hp)'] = err_p_based_on_larget_hp
    return df, pdf
    

def draw_paper_data(df, pdf, drop_col_names):
    hdf = df.drop(drop_col_names, axis=1)
    
    hdf = hdf.groupby(['deg', '#Refine'],as_index = False).first()

    hdf['L2 Error(h)'] = hdf['L2 Error(h)'].apply(lambda x: '%.4e' % x)
    hdf['Strain Energy'] = hdf['Strain Energy'].apply(lambda x: '%.6e' % x)
    print('L2 ERROR based on h-refinement:\n')
    print(hdf.to_latex(index=False))


    pdf =pdf.groupby(['#Refine','deg'],as_index = False).first()
    pdf = pdf.drop(drop_col_names, axis=1)
    pdf['L2 Error(p)'] = pdf['L2 Error(p)'].apply(lambda x: '%.4e' % x)
    pdf['Strain Energy'] = pdf['Strain Energy'].apply(lambda x: '%.6e' % x)
    print('L2 ERROR based on p:\n')
##    print('L2 ERROR based on p and hp-refinement:\n')
##    pdf['L2 Error(hp)'] = pdf['L2 Error(hp)'].apply(lambda x: '%.4e' % x)
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
        print('Not enough data recorded for:')
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
        print('Not enough data recorded for:')
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


def plot_stong_scaling(df,p,title,filename=None):
    fig, ax = plt.subplots()
    plt_marker = [ '*','o', '^', 'p']
    plt_linestyle = ['--g','-.r', ':b', '--k']
    for i in range(len(p)):
        mdof_s = np.array((df.where(df['deg']==p[i])['MDoFs/Sec'].dropna()))
        solv_t = np.array(df.where(df['deg']==p[i])['Solve Time(s)'].dropna())
        num_np = np.array(df.where(df['deg']==p[i])['np'].dropna())
        y = mdof_s/num_np
        x = solv_t
        ax.plot(x,y,plt_linestyle[i], marker=plt_marker[i], label='p{}'.format(p[i]))
    plt.title(title)
    plt.legend(ncol = 2, loc="upper right", shadow=True)
    plt.xlabel('Solve Time (s)')
    plt.ylabel('(MDof/Sec) / (#Processors)', rotation=90)
    #plt.grid()
    if filename:
        plt.savefig(filename)
    plt.show()


def plot_stong_scaling_np(df,p,title, xtick_vals, xticks_val_str, filename=None):
    fig, ax = plt.subplots()
    plt_marker = [ '*','o', '^', 'p']
    plt_linestyle = ['--g','-.r', ':b', '--k']
    for i in range(len(p)):
        mdof_s = np.array((df.where(df['deg']==p[i])['MDoFs/Sec'].dropna()))
        solv_t = np.array(df.where(df['deg']==p[i])['Solve Time(s)'].dropna())
        num_np = np.array(df.where(df['deg']==p[i])['np'].dropna())
        y = mdof_s/num_np
        x = num_np
        ax.semilogx(x,y,plt_linestyle[i], marker=plt_marker[i], label='p{}'.format(p[i])) #semilogx
    plt.title(title)
    plt.legend(ncol = 2, loc="upper right", shadow=True)
    plt.xticks(xtick_vals,xticks_val_str)
    plt.xlabel('#Processors')
    plt.ylabel('(MDof/Sec) / (#Processors)', rotation=90)
    plt.grid()
    if filename:
        plt.savefig(filename)
    plt.show()
    
     

if __name__ == "__main__":

    #log files' extension
    filename_ext = '.log'
    #number of repeats per simulation
    repeat = 3

    
                                              #Compressible Tube 
    #---------------------------------------------------------------------------------------------------
    print('-------------------------------Compressible-----------------------------------------------')
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
    roundFlag = True
    df, pdf = compute_error(df,refine,p,roundFlag)
    
    nu = 0.3
    ylim = [0.0001, 0.1]
    #---Pareto diagrams where L2 error is computed based on h-refienemnt---#
    x='Cost'
    y='L2 Error(h)'
    plot_cost_err_seaborn(df, x,y, 'error-h-cost-tube-comp.png',nu,ylim)
    x='Solve Time(s)'
    y='L2 Error(h)'
    plot_time_err_seaborn(df, x, y, 'error-h-time-tube-comp.png',nu,ylim)

    #---Pareto diagrams where L2 error is computed based on p-refienemnt---#
    x='Cost'
    y='L2 Error(p)'
    plot_cost_err_seaborn(pdf, x,y, 'error-p-cost-tube-comp.png',nu,ylim)
    x='Solve Time(s)'
    y='L2 Error(p)'
    plot_time_err_seaborn(pdf, x, y, 'error-p-time-tube-comp.png',nu,ylim)

    drop_col_names = ['#CG','MDoFs/Sec','Petsc Time(s)', 'Solve Time(s)','np']
    draw_paper_data(df,pdf, drop_col_names)
    h = [0.0030, 0.0020, 0.0015, 0.0012 ,0.0010]
    
    #draw_paper_data_tube(df,4) #<---- 4 mean use poly orders 1,2,3 and 4
    #---------------------------------------------------------------------------------------------------
    
    
                                            #Incompressible Tube
    #---------------------------------------------------------------------------------------------------
    print('-------------------------------Incompressible-----------------------------------------------')
    folder_name = 'log_files_tube_incomp'
    #indecies to keep from filename
    #idx:    0   1    2  3  4  5   6    7     8  9
    #     Tube8_20int_1_deg_3_cpu_384_incomp_run_2.log
    logfile_keywords = ['Global nodes', 'Total KSP Iterations', 'SNES Solve Time', 'DoFs/Sec in SNES', \
                        'Strain Energy', '.edu with','Time (sec):']
    keep_idx = [2,4,9]
    full_disp = True
    dof = 3
    df = process_log_files_linE_tube(folder_name, filename_ext, keep_idx, logfile_keywords,repeat, dof, full_disp)

    refine = [1,2,3,4,5]
    p = [2,3,4]
    roundFlag = False
    df, pdf = compute_error(df,refine,p,roundFlag)

    nu = 0.499999
    ylim = [.2, 1]
    #---Pareto diagrams where L2 error is computed based on h-refienemnt---#
    x='Cost'
    y='L2 Error(h)'
    plot_cost_err_seaborn(df, x,y, 'error-h-cost-tube-incomp.png',nu,ylim)
    x='Solve Time(s)'
    y='L2 Error(h)'
    plot_time_err_seaborn(df, x, y, 'error-h-time-tube-incomp.png',nu,ylim)

    #---Pareto diagrams where L2 error is computed based on p-refienemnt---#
    x='Cost'
    y='L2 Error(p)'
    plot_cost_err_seaborn(pdf, x,y, 'error-p-cost-tube-incomp.png',nu,ylim)
    x='Solve Time(s)'
    y='L2 Error(p)'
    plot_time_err_seaborn(pdf, x, y, 'error-p-time-tube-incomp.png',nu,ylim)
    
    #print(df)
    drop_col_names = ['#CG','MDoFs/Sec','Petsc Time(s)', 'Solve Time(s)','np']
    draw_paper_data(df,pdf, drop_col_names)
    h = [0.0030, 0.0020, 0.0015, 0.0012 ,0.0010]
 
    #draw_paper_data_tube(df,3)  #<---- 3 mean use poly orders 2,3 and 4
    #---------------------------------------------------------------------------------------------------

    

                                                   #Beam
    #---------------------------------------------------------------------------------------------------
    print('-------------------------------Beam-----------------------------------------------')
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
    roundFlag = False
    df, pdf = compute_error(df,refine,p,roundFlag)
    drop_col_names = ['#CG','MDoFs/Sec','Petsc Time(s)', 'Solve Time(s)','np']
    draw_paper_data(df,pdf, drop_col_names)

    print("slopes for beam with poly order 1-4 ")
    #plot_beam_p_conv(pdf, refine ,p)

    h = [0.1428, 0.0714, 0.0476 ,0.0357, 0.0286]
    plot_beam_h_conv(df, p ,h, 'beam_err_h.png')
    
    #cs = compute_conv_slope(df,h)
    #print(cs)
    
    #---------------------------------------------------------------------------------------------------

# Beam h sizes(Beam8 0-4 refinements): 0.1428    0.0714    0.0476    0.0357    0.0286 meters

# Tube h sizes (Tube8_20_{1-5} refinements): 0.0030    0.0020    0.0015    0.0012    0.0010 meters    
    


                                              #Compressible Tube-scaling 
    #---------------------------------------------------------------------------------------------------
    print('-------------------------------Compressible-----------------------------------------------')
    folder_name = 'scaling_log_files_comp'
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
    p = [1,2,3,4]
    title ='Throughput vs. Time (Compressible Tube Bend)'
    plot_stong_scaling(df,p, title,'throughput-time-tube-comp')
    title ='Throughput vs. #procs (Compressible Tube Bend)'
    xtick_vals = [12,24,48,96,192,384,768]
    xticks_val_str = ['12','24','48','96','192','384','768']
    plot_stong_scaling_np(df,p,title, xtick_vals, xticks_val_str ,'throughput-np-tube-comp')
##
    print('-------------------------------Incompressible-----------------------------------------------')
    folder_name = 'scaling_log_files_incomp'
    #indecies to keep from filename
    #idx:    0   1    2  3  4  5   6    7     8  9
    #     Tube8_20int_1_deg_3_cpu_384_incomp_run_2.log
    logfile_keywords = ['Global nodes', 'Total KSP Iterations', 'SNES Solve Time', 'DoFs/Sec in SNES', \
                        'Strain Energy', '.edu with','Time (sec):']
    keep_idx = [2,4,9]

    full_disp = True
    dof = 3
    df = process_log_files_linE_tube(folder_name, filename_ext, keep_idx, logfile_keywords,repeat,dof, full_disp)
    print(df)
    p = [2,3,4]
    title ='Throughput vs. Time (Incompressible Tube Bend)'
    plot_stong_scaling(df,p, title, 'throughput-time-tube-incomp')
    xtick_vals = [192,384,768,1536]
    xticks_val_str = ['192','384','768','1536']
    title ='Throughput vs. #np (Incompressible Tube Bend)'
    plot_stong_scaling_np(df,p,title, xtick_vals, xticks_val_str,'throughput-np-tube-incomp')
    

    

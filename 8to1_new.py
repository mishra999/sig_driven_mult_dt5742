import os.path

import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
records = 500000
n1 = np.zeros((500000, 1000))
n2 = np.zeros((500000, 1000))

        
with open(r'C:\Users\mmishra\OneDrive - North Carolina State University\drs4_multiplexerr\8to1mult\dt5742\wave_7.dat', 'rb') as f:
    ct = 0 
    while(1):
        b = np.fromfile(f, dtype=np.int32, count=6)
#        print('file header=',b)
        b1 = np.fromfile(f, dtype=np.float32, count=1024)
        n2[ct] = b1[15:1015] - np.mean(b1[15:95])
#        print('file header=',b)
        ct = ct + 1
        if ct > records-1:
            f.close()
            break


with open(r'C:\Users\mmishra\OneDrive - North Carolina State University\drs4_multiplexerr\8to1mult\dt5742\wave_0.dat', 'rb') as f:
    ct = 0 
    while(1):
        b = np.fromfile(f, dtype=np.int32, count=6)
#        print('file header=',b)
        b1 = np.fromfile(f, dtype=np.float32, count=1024)
        n1[ct] = b1[15:1015] - np.mean(b1[15:95])
#        print('file header=',b)
        ct = ct + 1
        if ct > records-1:
            f.close()
            break

from numpy import load
N1 = 1000
fs = 1000
f0 = np.arange(N1)
f0 = (fs * 1. /N1) * f0        
        
plt.figure()

plt.plot(n1[2073])
plt.plot(n2[2073])
plt.show()
plt.plot(f0,np.abs(np.fft.fft(n2[2073])))
plt.show()


res_50 = []
#res_55 = []
#res_85 = []
#res_65 = []
r_extra = []
for i in range(500000):
    res_temp50 = []
#    res_temp65 = []
#    res_temp60 = []
#    res_temp85 = []
    cnt = 0
    cnt50 = 0
    cnt55 = 0
    cnt60 = 0
    cnt65 = 0
    d1 = np.abs(np.fft.fft(n2[i]))
    if 40 + np.argmax(d1[40:95])== 48 or 40 + np.argmax(d1[40:95])== 49 or 40 + np.argmax(d1[40:95])== 50:
#    if 48 + np.argmax(d1[48:52]) == 49 or 48 + np.argmax(d1[48:52]) == 50:
#        if d1[48 + np.argmax(d1[48:52])] - d1[48 + np.argmax(d1[48:52]) - 1] > 0. and d1[48 + np.argmax(d1[48:52])] - d1[48 + np.argmax(d1[48:52]) - 2] > 5000.:
#            if d1[48 + np.argmax(d1[48:52])] - d1[48 + np.argmax(d1[48:52]) + 1] > 0. and d1[48 + np.argmax(d1[48:52])] - d1[48 + np.argmax(d1[48:52]) + 2] > 5000.:
        res_temp50.append(n2[i])
        res_temp50.append(n1[i])
        res_50.append(res_temp50)

        
#                cnt += 1
#                cnt50 += 1
    elif 10 + np.argmax(d1[10:500])== 54 or 10 + np.argmax(d1[10:500])== 55 or 10 + np.argmax(d1[10:500])== 56:            
#    if 53 + np.argmax(d1[53:57]) == 54 or 53 + np.argmax(d1[53:57]) == 55:
#        if d1[53 + np.argmax(d1[53:57])] - d1[53 + np.argmax(d1[53:57]) - 1] > 0. and d1[53 + np.argmax(d1[53:57])] - d1[53 + np.argmax(d1[53:57]) - 2] > 5000.:
#            if d1[53 + np.argmax(d1[53:57])] - d1[53 + np.argmax(d1[53:57]) + 1] > 0. and d1[53 + np.argmax(d1[53:57])] - d1[53 + np.argmax(d1[53:57]) + 2] > 5000.:
        res_temp55.append(a00[3,i])
        res_temp55.append(aa00[0,i])#SiPM siganla connected to second board
        res_55.append(res_temp55)


#                cnt += 1
#                cnt55 += 1
    elif 10 + np.argmax(d1[10:500])== 59 or 10 + np.argmax(d1[10:500])== 60 or 10 + np.argmax(d1[10:500])== 61:
#    if 58 + np.argmax(d1[58:62]) == 59 or 58 + np.argmax(d1[58:62]) == 60:
#        if d1[58 + np.argmax(d1[58:62])] - d1[58 + np.argmax(d1[58:62]) - 1] > 0. and d1[58 + np.argmax(d1[58:62])] - d1[58 + np.argmax(d1[58:62]) - 2] > 5000.:
#            if d1[58 + np.argmax(d1[58:62])] - d1[58 + np.argmax(d1[58:62]) + 1] > 0. and d1[58 + np.argmax(d1[58:62])] - d1[58 + np.argmax(d1[58:62]) + 2] > 5000.:
        res_temp60.append(a00[3,i])
        res_temp60.append(a00[1,i])
        res_60.append(res_temp60)


#                cnt += 1 
#                cnt60 += 1
    elif 10 + np.argmax(d1[10:500])== 64 or 10 + np.argmax(d1[10:500])== 65 or 10 + np.argmax(d1[10:500])== 66:            
#    if 63 + np.argmax(d1[63:67]) == 64 or 63 + np.argmax(d1[63:67]) == 65:
#        if d1[63 + np.argmax(d1[63:67])] - d1[63 + np.argmax(d1[63:67]) - 1] > 0. and d1[63 + np.argmax(d1[63:67])] - d1[63 + np.argmax(d1[63:67]) - 2] > 5000.:
#            if d1[63 + np.argmax(d1[63:67])] - d1[63 + np.argmax(d1[63:67]) + 1] > 0. and d1[63 + np.argmax(d1[63:67])] - d1[63 + np.argmax(d1[63:67]) + 2] > 5000.:
        res_temp65.append(a00[3,i])
        res_temp65.append(a00[2,i])#SiPM siganla connected to second board
        res_65.append(res_temp65)
    else:
        r_extra.append(a00[3,i])

res_50 = np.asarray(res_50)
res_55 = np.asarray(res_55)
res_60 = np.asarray(res_60)
res_65 = np.asarray(res_65)

plt.plot(f0,np.abs(np.fft.fft(res_50[27777,0])))

save(os.path.join(path +  '\\','res_50'), res_50)

res_50 = load(os.path.join(path +  '\\','res_50' +'.npy'))

from numpy import save

def DT5742_read(path, file_name, rlength, records, avg_length):
    n1 = np.zeros((records, rlength))
    wdir = os.path.join(path,file_name)
    with open(wdir, 'rb') as f:
        ct = 0 
        while(1):
            b = np.fromfile(f, dtype=np.int32, count=6)
    #        print('file header=',b)
            b1 = np.fromfile(f, dtype=np.float32, count=1024)
            n1[ct] = b1[15:1015] - np.mean(b1[15:1015]) 
    #        print('file header=',b)
            ct = ct + 1
            if ct > records-1:
                f.close()
                break 
    return n1

path = r'\Users\mmishra\OneDrive - North Carolina State University\drs4_multiplexerr\8to1mult\dt5742'
folders = ['50MHz', '55MHz', '60MHz', '65MHz', '70MHz', '75MHz', '80MHz', '85MHz']
files = ['wave_0.dat', 'wave_2.dat']

for i in range(1):#read data for 4 resonators to get impulse response
    for j in range(2):#read noise and fanin data
        dict_noise_input = {'path': path +  '\\' + folders[i] + '\\', 'file_name': files[j], 'rlength': 1000, 'records': 20000, 'avg_length': 750}
        if j ==0:
            noise = DT5742_read(**dict_noise_input)
            save(os.path.join(dict_noise_input['path'] +  '\\','noise'), noise)
            print(folders[i], files[j], ' done!')
    
        else:
            fanin = DT5742_read(**dict_noise_input)
            save(os.path.join(dict_noise_input['path'] +  '\\','fanin'), fanin)
            print(folders[i], files[j], ' done!')
            
def impulse_response (noise, fanin, rlength, records):
    #do correlation in time , take average in time, then dive in frequency domain
    from scipy import signal
    corr1 = np.zeros((records, 2*rlength - 1))
    corr2 = np.zeros((records, 2*rlength - 1))
    h1 = np.zeros(2*rlength - 1, dtype=np.complex)
    h2 = np.zeros(2*rlength - 1, dtype=np.complex)
    h = np.zeros(2*rlength - 1,dtype=np.complex)
    for i in range(records):
        corr1[i] = signal.correlate(fanin[i], noise[i], mode='full')# / (2*rlength - 1)
        corr2[i] = signal.correlate(noise[i], noise[i], mode='full')# / (2*rlength - 1)
    
    #average in time
    corrr1 =  np.zeros(2*rlength - 1)
    corrr2 =  np.zeros(2*rlength - 1)
    for i in range(2*rlength - 1):
        for j in range(records):
            if j == 0:
                corrr1[i] = corr1[j,i]
                corrr2[i] = corr2[j,i]
            else:
                corrr1[i] = 1/(j+1) * (corr1[j,i] + j*corrr1[i])
                corrr2[i] = 1/(j+1) * (corr2[j,i] + j*corrr2[i])
#            corrr1[i] = corrr1[i] + corr1[j,i]
#            corrr2[i] = corrr2[i] + corr2[j,i]    
#    corrr1 = corrr1/records
#    corrr2 = corrr2/records
    #tafe fft of average in frequency domain
    h1 = np.fft.fft(corrr1)
    h2 = np.fft.fft(corrr2)
    #divide cross-correlation by autocorrelation
    h = h1 / h2
    #get impulse response in time
    h_data = np.real(np.fft.ifft(h))
    
    return h_data, corrr1, corrr2



from numpy import load
N1 = 1000
fs = 1000
f0 = np.arange(N1)
f0 = (fs * 1. /N1) * f0

N1 = 1999
fs = 1000
f00 = np.arange(N1)
f00 = (fs * 1. /N1) * f00
    
for i in range(1):    
    noise = load(os.path.join(path +  '\\' + folders[i] + '\\','noise.npy'))
    fanin = load(os.path.join(path +  '\\' + folders[i] + '\\','fanin.npy'))
    dict_impulse_res = {'noise': noise, 'fanin': fanin, 'rlength': 1000, 'records': 20000}
    imp_res, corrr1, corrr2 = impulse_response(**dict_impulse_res)
    save(os.path.join(path +  '\\' + folders[i] + '\\','impres'+folders[i]), imp_res)
    plt.figure()
    if i <3:
        plt.plot(f0[1:500],np.abs(np.fft.fft(imp_res[0:1000]))[1:500], label = folders[i][0:2]+' MHz')
    else:
        plt.plot(f0[0:500],np.abs(np.fft.fft(imp_res[0:1000]))[0:500], label = folders[i][0:2]+' MHz')
    plt.ylabel('amplitude (arb. units)', fontsize=16)
    plt.xlabel('frequency (MHz)', fontsize = 16)
    plt.legend()
    plt.tight_layout()
    plt.show()
    # plt.figure()
    # plt.plot(f00[0:1000],np.abs(np.fft.fft(corrr1))[0:1000])
    # plt.plot(f00[0:1000],np.abs(np.fft.fft(corrr2))[0:1000])
    savefig(os.path.join(path +  '\\' + folders[i] + '\\','impres' + folders[i]))
    # plt.close()

def get_impulse_response(path, folder, resonator):
    return load(os.path.join(path +  '\\' + folder +  '\\','impres' + resonator +'.npy'))

resonators = ['50MHz', '7MHz', '9MHz', '11MHz']
impres = []
for i in range(1):
    impres.append(get_impulse_response(path, folders[i], resonators[i]))
imp_response = dict(zip(resonators, impres))
    
plt.plot(imp_response['50MHz'][0:1000])

plt.ion()     
    
dict_impulse_res = {'noise': n1, 'fanin': n2, 'rlength': 1000, 'records': 20000}
imp_res = impulse_response(**dict_impulse_res)
from collections import deque
imp = deque(imp_res)
imp1 = imp.rotate(1)
plt.plot(imp)


from numpy import load
N1 = 1000
fs = 1000
f0 = np.arange(N1)
f0 = (fs * 1. /N1) * f0
plt.figure()
#plt.plot(imp_res[0:1000])
plt.plot(f0[0:500],np.abs(np.fft.fft(imp_response['50MHz'][0:1000]))[0:500], label ='50MHz')
plt.xlabel('frequency (MHz)', fontsize=14)
plt.ylabel('amplitude (arb. units)', fontsize=14)
plt.legend()
plt.tight_layout()

#get amp and time

#50 MHz

      
import numpy as np
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt
import scipy
from scipy import signal
from scipy.interpolate import UnivariateSpline
N, Wn = signal.buttord((200/500),(250/500) , 3, 10) #(240/500),(290/500) , 3, 10, charge        (130/500),(170/500) , 3, 10, timing
b, a = signal.butter(N, Wn, 'low')

#h50_f = np.fft.fft(h50_t1[0:1050])
#rec_50 = np.zeros((len(res_50),1050))
#for i in range(len(res_50)):
#    out_x =  np.fft.fft(np.lib.pad(res_50[i,0], (0,50), 'constant', constant_values=(0., 0.)))/h50_f
#    out_xn = np.real(np.fft.ifft(out_x))
#    rec_50[i] = scipy.signal.filtfilt(b, a, out_xn)
    
h50_f = np.fft.fft(imp_response['50MHz'][0:1000])
rec_50 = np.zeros((len(res_50),1000))
for i in range(len(res_50)):
    out_x =  np.fft.fft(res_50[i,0])/h50_f
    out_xn = np.real(np.fft.ifft(out_x))
    rec_50[i] = scipy.signal.filtfilt(b, a, out_xn) - np.mean(scipy.signal.filtfilt(b, a, out_xn)[5:95])

for i in range(90,91):
    plt.plot(rec_50[i])
    plt.plot(res_50[i,0])
    plt.plot(res_50[i,1])

plt.plot(f0[0:500],np.abs(np.fft.fft(rec_50[i]))[0:500])
plt.plot(f0[0:500],np.abs(np.fft.fft(res_50[i,1]))[0:500])
plt.plot(f0[0:500],np.abs(np.fft.fft(res_50[i,0]))[0:500])

plt.show()

import copy
g_res50=[]
rres_50 = copy.deepcopy(res_50)
rrec_50 = copy.deepcopy(rec_50)
t_ind = []
t_res50 = []
from scipy.interpolate import UnivariateSpline
max_res50 = np.zeros((len(rres_50),2))
rew = 0
for i in range(len(rec_50)):#len(rec_50)
    max_arg1 = 100 + np.argmax(rres_50[i,1,100:900])
    abcissa1 = np.asarray([ind for ind in range(max_arg1-4, max_arg1+5)])
    ordinate1 = rres_50[i,1,abcissa1]
    spl1 = UnivariateSpline(abcissa1, ordinate1,k=3)
    xs1 = np.linspace(max_arg1-4, max_arg1+5,80)

    max_res50[i,0] = np.max(spl1(xs1)[2:-2])
#    plt.plot(abcissa1, ordinate1)
#    plt.plot(xs1, spl1(xs1))
#    plt.show()
#    warnings.simplefilter('error', UserWarning)
#    max_res50[i,0] = np.max(rres_50[i,1,100:900])

    max_arg2 = 100 + np.argmax(rrec_50[i, 100:900])
    abcissa2 = np.asarray([ind for ind in range(max_arg2-4, max_arg2+5)])
    ordinate2 = rrec_50[i,abcissa2]
    spl2 = UnivariateSpline(abcissa2, ordinate2,k=3)
    xs2 = np.linspace(max_arg2-4, max_arg2+5,80)
    max_res50[i,1] = np.max(spl2(xs2)[2:-2])
#    max_res50[i,1] = np.max(rec_50[i, 100:900])
    
    
    if max_res50[i,1] < 24765.:#80 keV
        t_ind.append(i)
        g_temp =[]
        g_temp.append(max_res50[i,0])
        g_temp.append(max_res50[i,1])
        g_res50.append(g_temp)
        
        
        t_temp = []
        for ii in range(len(abcissa1)-3):
            if ordinate1[ii] <= max_res50[i,0]/2 < ordinate1[ii+1]:
                yy1 = ordinate1[ii]
                xx1 = abcissa1[ii]
                yy2 = ordinate1[ii+1]
                xx2 = abcissa1[ii+1]
                break
        if ii==5:
            t_temp.append(2. )
            t_temp.append(-2. )
            t_res50.append(t_temp)
            continue
        slope = (yy1 - yy2) / (xx1 - xx2)
        intrcept = yy1 - (slope*xx1) 
        t_temp.append(((max_res50[i,0]/2) - intrcept)/slope )
    
        for ii2 in range(len(abcissa2)-3):
            if ordinate2[ii2] <= max_res50[i,1]/2 < ordinate2[ii2+1]:
                yy1 = ordinate2[ii2]
                xx1 = abcissa2[ii2]
                yy2 = ordinate2[ii2+1]
                xx2 = abcissa2[ii2+1]
                break

        if ii2==5:
            t_temp.append(20. )
#            t_temp.append(-2. )
            t_res50.append(t_temp)
            continue
                
        slope = (yy1 - yy2) / (xx1 - xx2)
        intrcept = yy1 - (slope*xx1) 
        t_temp.append(((max_res50[i,1]/2) - intrcept)/slope )
            
        t_res50.append(t_temp)
        


g_res50 = np.asarray(g_res50)#28653
t_res50 = np.asarray(t_res50)



#check if different fit rangesusing cubic spline for the pulse does anything? Nothing! 9050
fig7 = plt.figure()
nbins = 306
hq, bnedgess  = np.histogram(g_res50[:,1],bins=np.arange(50, 2500, 8))
plt.hist(g_res50[:,1], bins=np.arange(100, 2500, 8), histtype = 'step')
yxq = 0.8*np.max(hq[110:170])*np.ones(nbins)
yxq1 = 391.72*0.8*np.ones(nbins)
bne11=(bnedgess[1:]+bnedgess[:-1])/2
#plt.plot(bne11,yxq)
plt.plot(bne11,yxq1)#764
#11560/477.3*80
hq, bnedgess  = np.histogram(g_res50[:,0],bins=np.arange(50, 2500, 8))
plt.hist(g_res50[:,0], bins=np.arange(100, 2500, 8), histtype = 'step')
yxq = 0.8*np.max(hq[110:170])*np.ones(nbins)
#yxq1 = 0.8*250.8*np.ones(169)
bne11=(bnedgess[1:]+bnedgess[:-1])/2
plt.plot(bne11,yxq)
#plt.plot(bne11,yxq1)#15000
#11820/477.3*80

plt.xlabel('charge collected \n(arb. units)',fontsize=16)
plt.ylabel('frequency',fontsize=16)
plt.tight_layout()
plt.show()

plt.scatter(g_res50[0:3000,1]/2**12*1000, g_res50[0:3000,0]/2**12*1000,s=0.1, c='k', alpha=0.2)
plt.xlabel('recovered peak (mV)')#r'\textbf{time}
plt.ylabel('original peak (mV)')

plt.figure()
plt.hist2d((g_res50[:,0]-g_res50[:,1]), g_res50[:,1], cmin = 1, bins=1000)
plt.xlabel('original peak - recovered peak \n (mV)')
plt.ylabel('recovered peak \n (mV)')
plt.tight_layout()

diffa_res50 = []
difft_res50 = []
gg_res50 = []

for i in range(len(g_res50[:,0])):
    if g_res50[i,0]-g_res50[i,1] > -20:
        diffa_temp = []
        difft_temp = []
        gg_res50_temp = []
        diffa_temp.append(g_res50[i,0]-g_res50[i,1])
        diffa_temp.append(g_res50[i,1])
        diffa_res50.append(diffa_temp)
        difft_temp.append(t_res50[i,0]-t_res50[i,1]-2)
        difft_temp.append(g_res50[i,1])
        difft_res50.append(difft_temp)
        gg_res50_temp.append(g_res50[i,0])
        gg_res50_temp.append(g_res50[i,1])
        gg_res50.append(gg_res50_temp)
#        plt.plot(res_50[i,1])
#        plt.plot(rec_50[i])
#        break

diffa_res50 =  np.asarray(diffa_res50)
gg_res50 = np.asarray(gg_res50)
plt.figure()
plt.hist2d(diffa_res50[:,0]/2**12*1000, diffa_res50[:,1]/2**12*1000, bins=[np.arange(-20/2**12*1000, 120/2**12*1000, 1/2**12*1000), np.arange(30/2**12*1000, 1000/2**12*1000, 5/2**12*1000)], cmin = 1)
plt.xlabel('original peak - recovered peak \n (mV)')
plt.ylabel('recovered peak \n (mV)')
plt.tight_layout()

plt.plot(f0[0:500],np.abs(np.fft.fft(res_50[i,0]))[0:500])
plt.plot(f0[0:500],np.abs(np.fft.fft(res_50[i,1]))[0:500])
plt.plot(f0[0:500],np.abs(np.fft.fft(rec_50[i]))[0:500])


difft_res50 =  np.asarray(difft_res50)

plt.figure()
plt.hist2d(difft_res50[:,0]*1000, difft_res50[:,1]/2**12*1000, bins=[np.arange(-1500, 700, 10), np.arange(30/2**12*1000, 1000/2**12*1000, 5/2**12*1000)], cmin = 1)
plt.xlabel('original peak - recovered peak \n (mV)')
plt.ylabel('recovered peak \n (mV)')
plt.tight_layout()

plt.plot(res_50[8008,1], label='original signal')
plt.plot(res_50[8008,0], label='freq-encoded signal')
plt.xlabel('sample number',fontsize=14)
plt.ylabel('sample value \n(ADC units)',fontsize=14)  
plt.legend()      
plt.tight_layout()

plt.plot(res_50[i,0])
plt.xlabel('sample number',fontsize=16)
plt.ylabel('sample value (ADC units)',fontsize=16)        
plt.tight_layout()

plt.plot(f0[0:500],np.abs(np.fft.fft(res_50[8008,1]))[0:500], label='original signal')
plt.plot(f0[0:500],np.abs(np.fft.fft(res_50[8008,0]))[0:500], label='freq-encoded signal')
#plt.plot(f0[0:500],np.abs(np.fft.fft(rec_50[8008]))[0:500])
plt.xlabel('frequency (MHz)', fontsize=16)
plt.ylabel('amplitude\n(arb. units)', fontsize=16)
plt.legend()      
plt.tight_layout()


for i11 in range(len(res_50)):
    if 800<np.max(res_50[i11,1,100:900])<900:
        
        plt.figure()
        plt.plot(res_50[i11,1])
        plt.plot(rec_50[i11])
        plt.show()
        print(i11)#2538
        break
    
plt.plot(np.abs(np.fft.fft(res_50[i11,0])))
plt.plot(np.abs(np.fft.fft(rec_50[i11])))

for i22 in range(500,len(res_50)):
    if 140<np.max(res_50[i22,1,100:900])<160:
        
        plt.figure()
        plt.plot(res_50[i22,1])
        plt.plot(rec_50[i22])
        plt.show()
        print(i22)#2538
        break
plt.plot(np.abs(np.fft.fft(rec_50[i33])))
plt.plot(np.abs(np.fft.fft(res_50[i33,1])))


fig, axs = plt.subplots(2,2)
axs[0,0].plot(res_50[i22,1,100 + np.argmax(res_50[i22,1,100:900]) - 12 : 100 + 
            np.argmax(res_50[i22,1,100:900]) + 30], alpha = 0.75,label='original pulse')
axs[0,0].legend(fontsize= 14)

axs[0,0].plot(rec_50[i22,100 + np.argmax(res_50[i22,1,100:900]) - 14 : 100 + 
            np.argmax(res_50[i22,1,100:900]) + 28], alpha = 0.75,label='recovered pulse')
axs[0,0].legend()
axs[0,1].plot(res_50[i22,1,100 + np.argmax(res_50[i22,1,100:900]) - 12 : 100 + 
            np.argmax(res_50[i22,1,100:900]) + 30] - rec_50[i22,100 + np.argmax(res_50[i22,1,100:900]) - 14 : 100 + 
            np.argmax(res_50[i22,1,100:900]) + 28], alpha = 0.7,label='residual',color='g')
axs[0,1].legend()
axs[0,1].set_ylim(-100,150)
axs[1,0].plot(res_50[i11,1,100 + np.argmax(res_50[i11,1,100:900]) - 12 : 100 + 
            np.argmax(res_50[i11,1,100:900]) + 30], alpha = 0.75,label='original pulse')
axs[1,0].legend()
axs[1,0].plot(rec_50[i11,100 + np.argmax(res_50[i11,1,100:900]) - 14 : 100 + 
            np.argmax(res_50[i11,1,100:900]) + 28], alpha = 0.75,label='recovered pulse')
axs[1,0].legend()
axs[1,1].plot(res_50[i11,1,100 + np.argmax(res_50[i11,1,100:900]) - 12 : 100 + 
            np.argmax(res_50[i11,1,100:900]) + 30] - rec_50[i11,100 + np.argmax(res_50[i11,1,100:900]) - 14 : 100 + 
            np.argmax(res_50[i11,1,100:900]) + 28], alpha = 0.7,label='residual',color='g')
#axs[1,1].set_ylim(750,-640)
axs[0,0].tick_params(axis="x", labelsize=7)
axs[0,0].tick_params(axis="y", labelsize=7)
axs[0,1].tick_params(axis="x", labelsize=7)
axs[0,1].tick_params(axis="y", labelsize=7)
axs[1,0].tick_params(axis="x", labelsize=7)
axs[1,0].tick_params(axis="y", labelsize=7)
axs[1,1].tick_params(axis="x", labelsize=7)
axs[1,1].tick_params(axis="y", labelsize=7)
plt.legend()
fig.text(0.5, 0.01, 'sample number (ns)', ha='center',size= 14)
#plt.xlabel("sample number (ns)")
fig.text(0.03,0.5, "sample value (ADC units)", ha="center", va="center", rotation=90,size= 14)
plt.suptitle('50 MHz resonator',size= 16)
plt.show()

#check if different fit rangesusing cubic spline for the pulse does anything? Nothing! 9050
fig7 = plt.figure()
nbins = 306
hq, bnedgess  = np.histogram(gg_res50[:,1],bins=np.arange(50, 2500, 8))
plt.hist(gg_res50[:,1], bins=np.arange(100, 2500, 8), histtype = 'step')
yxq = 0.8*np.max(hq[110:170])*np.ones(nbins)
yxq1 = 391.72*0.8*np.ones(nbins)
bne11=(bnedgess[1:]+bnedgess[:-1])/2
#plt.plot(bne11,yxq)
plt.plot(bne11,yxq1)#812
#812/477.3*80
hq, bnedgess  = np.histogram(gg_res50[:,0],bins=np.arange(50, 2500, 8))
plt.hist(gg_res50[:,0], bins=np.arange(100, 2500, 8), histtype = 'step')
yxq = 0.8*np.max(hq[110:170])*np.ones(nbins)
#yxq1 = 0.8*250.8*np.ones(169)
bne11=(bnedgess[1:]+bnedgess[:-1])/2
plt.plot(bne11,yxq)
#plt.plot(bne11,yxq1)#15000
#11820/477.3*80

plt.xlabel('charge collected \n(arb. units)',fontsize=16)
plt.ylabel('frequency',fontsize=16)
plt.tight_layout()
plt.show()

plt.figure()
plt.hist2d(diffa_res50[:,0]*477.3/812, diffa_res50[:,1]*477.3/812, bins=[np.arange(-40*477.3/812, 120*477.3/812, 1*477.3/812), np.arange(30*477.3/812, 1000*477.3/812, 5*477.3/812)], cmin = 1)
plt.xlabel('original peak - recovered peak \n (keV)')
plt.ylabel('recovered peak \n (keV)')
plt.tight_layout()


fig, ax1 = plt.subplots(constrained_layout=True, figsize=(6,5))
ax1.tick_params(axis='both', which='major', labelsize=11)
cmap = mpl.colors.LinearSegmentedColormap.from_list("", ["black","pink","red"])
counts, xedges, yedges, im = ax1.hist2d((g_res50[:,0]-g_res50[:,1])*477.3/11820, g_res50[:,1]*477.3/11820 , bins=[np.arange(-30, 40, 0.25), np.arange(80, 1000, 5)],  cmap=cmap, cmin = 1)#, cmin = 1
ax2 = ax1.twinx()
ax2.tick_params(axis='both', which='major', labelsize=11)
mn, mx = ax1.get_ylim()
ax2.set_ylim(mn/477.3*11820/2**16*1000, mx/477.3*11820/2**16*1000)
ax2.set_ylabel('(mV)', fontsize = 14)
ax1.set_xlabel('original peak - recovered peak \n (keV)', fontsize = 13)
ax1.set_ylabel('recovered peak \n (keV)', fontsize = 14)
#plt.tight_layout()
#plt.xlim(-30,20)
cbar = fig.colorbar(im, ax=ax1)
cbar.ax.tick_params(labelsize=11)

plt.show()

plt.figure()
plt.hist2d(difft_res50[:,0]*1000, diffa_res50[:,1]*477.3/812, bins=[np.arange(-1500, 700, 10), np.arange(30*477.3/812, 1000*477.3/812, 5*477.3/812)], cmin = 1)
plt.xlabel('original timing - recovered timing \n (ps)')
plt.ylabel('recovered peak \n (keV)')
plt.tight_layout()

plt.figure()
plt.hist2d(diffa_res50[:,0]*477.3/812, diffa_res50[:,1]*477.3/812, bins=[np.arange(-40*477.3/812, 120*477.3/812, 1*477.3/812), np.arange(30*477.3/812, 1000*477.3/812, 5*477.3/812)], cmin = 1)
plt.xlabel('original peak - recovered peak \n (keV)')
plt.ylabel('recovered peak \n (keV)')
plt.tight_layout()

e_range = [80., 120.,150., 200., 300.,400., 500.]
diff_amp50 = [[] for i in range(6)]
diff_t50 = [[] for i in range(6)]

for i in range(len(e_range)-1):
    for j in range(len(gg_res50[:,1])):
        if e_range[i] < gg_res50[j,1]*477.3/812 < e_range[i+1]:
            diff_amp50[i].append((gg_res50[j,0] - gg_res50[j,1])*477.3/812)
            diff_t50[i].append((t_res50[j,0] - t_res50[j,1]-2.)*1000)

diff_ampp50 = [[] for i in range(6)]
for i in range(len(e_range)-1):
    for j in range(len(diff_amp50[i])):
        if i < 4:
            if np.mean(diff_amp50[i])-30 < diff_amp50[i][j] < np.mean(diff_amp50[i])+30:
                diff_ampp50[i].append(diff_amp50[i][j])
        else:
            if np.mean(diff_amp50[i])-50 < diff_amp50[i][j] < np.mean(diff_amp50[i])+50:
                diff_ampp50[i].append(diff_amp50[i][j])            

for i in range(len(e_range)-1):
    print('mean i:', np.mean(diff_ampp50[i]))
    print('std i:', np.sqrt(np.var(diff_ampp50[i])))
#    print('mean ti:', np.mean(diff_tt50[i]))
#    print('std ti:', np.sqrt(np.var(diff_tt50[i])))
    
#    plt.hist(diff_amp50[i], bins=60, color='gray')
#    plt.xlim(np.mean(diff_amp50[i])-23, np.mean(diff_amp50[i])+23)
    textstr = '\n'.join((
        r'$\mu=%.1f$ keV' % (np.mean(diff_ampp50[i]), ), 
        r'$\sigma=%.1f$ keV' % ( np.sqrt(np.var(diff_ampp50[i])), )))

    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', alpha=0.5)
    
    # place a text box in upper left in axes coords

    if i <6:
        fig, ax = plt.subplots()
        ax.hist(diff_ampp50[i], bins=np.arange(np.mean(diff_ampp50[i])-30, np.mean(diff_ampp50[i])+30, 1), color='gray')
        plt.xlabel('original peak - recovered peak \n (keV)')


    ax.text(0.02, 0.85, textstr, fontsize=16,
             transform=ax.transAxes)#, bbox=props
    plt.ylabel('counts')
    plt.show()
    savefig(r'C:\Users\mmishra\OneDrive - North Carolina State University\dissertation\final_chapters\Chapter5\paper4\charge60\charge50_' + str(int(e_range[i+1]) ))

diff_tt50 = [[] for i in range(6)]
for i in range(len(e_range)-1):
    for j in range(len(diff_t50[i])):
        if -1500 < diff_t50[i][j] < 700:
            diff_tt50[i].append(diff_t50[i][j])
        

for i in range(len(e_range)-1):
#    print('mean i:', np.mean(diff_amp50[i]))
#    print('std i:', np.sqrt(np.var(diff_amp50[i])))
    print('mean ti:', np.mean(diff_tt50[i]))
    print('std ti:', np.sqrt(np.var(diff_tt50[i])))
    
#    plt.hist(diff_amp50[i], bins=60, color='gray')
#    plt.xlim(np.mean(diff_amp50[i])-23, np.mean(diff_amp50[i])+23)
    textstr = '\n'.join((
        r'$\mu=%.1f$ ps' % (np.mean(diff_tt50[i]), ), 
        r'$\sigma=%.1f$ ps' % ( np.sqrt(np.var(diff_tt50[i])), )))

    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', alpha=0.5)
    
    # place a text box in upper left in axes coords

    if 0<=i <=3:
        fig, ax = plt.subplots()
        ax.hist(diff_tt50[i], bins=35, color='gray')
        plt.xlim(np.mean(diff_tt50[i])-1200, np.mean(diff_tt50[i])+1200)
        plt.xlabel('original timing - recovered timing (ps)')
        plt.show()
    elif i ==4 or i==5:
        fig, ax = plt.subplots()
        ax.hist(diff_tt50[i], bins=50, color='gray')
        plt.xlim(np.mean(diff_tt50[i])-1200, np.mean(diff_tt50[i])+1200)
        plt.xlabel('original timing - recovered timing (ps)')
        plt.show()
#    elif i ==5:
#        fig, ax = plt.subplots()
#        ax.hist(diff_tt60[i], bins=60, color='gray')
#        plt.xlim(np.mean(diff_tt60[i])-210, np.mean(diff_tt60[i])+210)
#        plt.xlabel('original timing - recovered timing (ps)')

#    elif i ==1 or i==2:
#        fig, ax = plt.subplots()
#        ax.hist(diff_t50[i], bins=35, color='gray')
#        plt.xlim(np.mean(diff_t50[i])-380, np.mean(diff_t50[i])+380)
#    else:
#        fig, ax = plt.subplots()
#        ax.hist(diff_tt60[i], bins=58, color='gray')#, label = 'mean: '+str(np.mean(diff_t50[i]))
#        plt.xlim(np.mean(diff_tt60[i])-210, np.mean(diff_tt60[i])+210)
#        plt.xlabel('original timing - recovered timing (ps)')
##        plt.legend(fontsize = 12)
        
    ax.text(0.02, 0.85, textstr, fontsize=16,
             transform=ax.transAxes)#, bbox=props
    plt.ylabel('counts')
    plt.show()
    
    
from scipy import optimize

def gaussian(x, amplitude, mean, stddev):
    return amplitude/ (np.sqrt(2*np.pi)*stddev) * np.exp(-(x - mean)**2 / (2*stddev**2) )

def find_single_pulse_mean_var(det_pulses, fdm_pulses, bin_width):
    bin_heights, bin_borders, _ = plt.hist(fdm_pulses/det_pulses, bins = np.arange(0, 500, bin_width))
    bin_centers = bin_borders[:-1] + np.diff(bin_borders) / 2
    
    for i in range(np.argmax(bin_heights)):
        if bin_heights[i] < np.max(bin_heights)/2 < bin_heights[i+1]:
            bin_l = i
    for i in range(np.argmax(bin_heights), len(bin_heights)-1):
        if bin_heights[i+1] < np.max(bin_heights)/2 < bin_heights[i]:
            bin_r = i    
    popt, pcov = optimize.curve_fit(gaussian, bin_centers[bin_l-2:bin_r+3], bin_heights[bin_l-2:bin_r+3], p0=[np.max(bin_heights)*(np.sqrt(2*np.pi)*(bin_centers[bin_r] - bin_centers[bin_l])/2.35), bin_centers[np.argmax(bin_heights)], (bin_centers[bin_r] - bin_centers[bin_l])/2.35])
    plt.plot(bin_centers[bin_l-2:bin_r+3],gaussian(bin_centers[bin_l-2:bin_r+3],*popt))
    plt.xlabel('pulse amplitude(frequency) / pulse amplitude(time)')
    plt.ylabel('counts')
    plt.xlim(0,100)

    return popt, pcov

dic_mean_var4to1 = {'det_pulses':data_4to1_amp, 'fdm_pulses':data_4to1_freq, 'bin_width':0.2}
#dic_mean_var2 = {'det_pulses':data_4to1_amp[1], 'fdm_pulses':coin_data_freq[1], 'bin_width':0.2}

popt1, pcov1 = find_single_pulse_mean_var(**dic_mean_var4to1)
print(popt1)
#popt2, pcov2 = find_single_pulse_mean_var(**dic_mean_var2)

pulse_filtered1 = []
#coin_pulse_filtered2 = []
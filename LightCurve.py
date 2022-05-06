#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 16:02:49 2019

@author: edu
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import math 
#from scipy import signal

class LightCurve:
  filename = ''
  t_sec = []
  y_ori = []
  FLAG_DFT = False
  timing = False
  n = 0
  name = ''
  filename = ''
  Amed = -1
  
  def __init__(self):
      pass
  
  def setTiming(self, Tstart=0., Tend=3600.*7, dT=10.):
      self.t_sec = np.arange(Tstart, Tend, dT, dtype='double')
      self.timing = True
      self.n = len(self.t_sec)
      self.y_ori = self.t_sec * 0.
      self.T = np.max(self.t_sec) - np.min(self.t_sec)
      self.fResolution = 1 / self.T 
      self.Amed = -1
      
  def addSinusoidalSignal(self, freq, amplit, phase):
      theta = 2. * np.pi * freq * self.t_sec 
      theta += phase * np.pi / 180.
      self.y_ori += amplit * np.sin(theta)
      
  def readDataFile(self, filex):
      x, y = np.loadtxt(filex, unpack=True)
      self.t_sec = x
      self.y_ori = y
      self.filename = filex 
      self.name = os.path.basename(filex)
      self.T = max(x) - min(x)
      self.fResolution = 1 / self.T 
      self.Amed = -1.
      self.timing = True


  def plot(self, ut='sec'):
      t = self.t_sec 
      xlab = 'Time (s)'
      if ut == 'hours':
          t = self.t_sec / 3600
          xlab = 'Time (hours)'
      if ut == 'days':
          t = self.t_sec / 86400
          xlab = 'Time (days)'
          
      plt.plot(t, self.y_ori)
      plt.title("Light Curve: " + self.name)
      plt.xlabel(xlab)
      plt.ylabel("Fractional Intensity")
      plt.show() 
      
      
  def summary(self):
      tmin = min(self.t_sec)
      tmax = max(self.t_sec)
      dT = tmax-tmin 
      
      ymin = min(self.y_ori)
      ymax = max(self.y_ori)
      
      print([tmin,tmax,dT])
      print([ymin,ymax])
      
      
  def plotdft(self, FREQ, AMPLIT, ufreq, uampl, limit):
      
      if ufreq=='Hz':
          f = FREQ
          xlab = 'Frequency (Hz)'
      if ufreq=='mHz':
          f = FREQ * 1000.
          xlab = 'Frequency (mHz)'
      if ufreq=='uHz':
          f = FREQ * 1.e6
          xlab = 'Frequency (uHz)'
          
      if uampl=='ma':
          a = AMPLIT
          ylab = 'Amplitude (ma)'
          amed = self.Amed
          
      if uampl=='mma':
          a = AMPLIT * 1000.
          ylab = 'Amplitude (mma)'
          amed = self.Amed * 1000.
          
      plt.plot(f, a)
      plt.title("Periodogram of " + self.name)
      plt.xlabel(xlab)
      plt.ylabel(ylab)
      
      #hlines(y, xmin, xmax, 
      #       colors='k', linestyles='solid', 
      #       label='', *, data=None, **kwargs)[source]
      if limit:
         xmin = min(f)
         xmax = max(f)
         plt.hlines(3.*amed, xmin, xmax, colors='red')
         plt.hlines(4.*amed, xmin, xmax, colors='green')
         
      plt.show() 
      
      
  def dft(self, fmin=0, fmax=3000.e-6, df=-1, 
          ufreq='Hz', uampl='ma', 
          plot=True, 
          amed=True,
          limit=False):
      
      if df==-1:
          df = self.fResolution / 10.
          
      t = self.t_sec
      y = self.y_ori
      n = len(t) 

      FREQ = np.arange(fmin, fmax, df, dtype=float)
      nf = len(FREQ)
      AMPLIT = np.repeat(0., nf)
      
      for k in range(0,nf):
          f = FREQ[k]
          w = 2. * np.pi * f
          FR = 0.
          FI = 0.
          for i in range(0,n):
              theta = w * t[i]
              FR += y[i] * np.cos(theta)
              FI += y[i] * np.sin(theta)
              #print(FR)
          FF = FR**2 + FI**2
          AMPLIT[k] = 2. * math.sqrt(FF) / nf
          
      if self.Amed<0 and amed:
          self.detectionLimit(AMPLIT, modo=1)
          
      if plot:
          self.plotdft(FREQ, AMPLIT, ufreq, uampl, limit)
          
      self.FREQ = FREQ
      self.AMPLIT = AMPLIT 
          

      return FREQ, AMPLIT
          
  def detectionLimit(self, A=[], modo=1): 
      #print("Calculating the detection limit...")
      
      if modo==1:
        n=len(A)
        npeaks = 0.
        Asum = 0.
      
        for i in range(1,n-1):
           if (A[i-1]>=A[i] and A[i]<A[i+1]): 
              npeaks += 1.
              Asum += A[i]
              
        self.Amed = Asum / npeaks          
      
      if modo==2:
        nsample = 200
      
        fmin = 3000.e-6
        fmax = self.fResolution * nsample
        df = self.fResolution / 10.
      
        f, A = self.dft(fmin, fmax, df, plot=False, amed=False)
        #print(A)
        n = len(A)
        npeaks = 0.
        Asum = 0.
      
        for i in range(1,n-1):
            if (A[i-1]<=A[i] and A[i]>=A[i+1]): 
                npeaks += 1.
                Asum += A[i]
              
        self.Amed = Asum / npeaks
      
      
  def getPeaks(self, amin=0., amed=True,
               uampl='ma', ufreq='Hz'):
      n = len(self.AMPLIT)
      fPeak = []
      APeak = [] 
      A = self.AMPLIT
      alim = amin
      
      if amed:
          alim = alim * self.Amed
          
      cf = 1.
      if ufreq=='mHz':
          cf = 1000.
      if ufreq=='uHz':
          cf = 1.e6
          
      ca = 1.
      if uampl=='mma':
          ca = 1000.
      
      for i in range(1,n-1):
          if A[i-1]<=A[i] and A[i]>=A[i+1]:
              if A[i] >= alim:
                  fPeak.append(self.FREQ[i] )
                  APeak.append(self.AMPLIT[i])
              
      npeaks = len(fPeak)
      print("\n Peaks > ", alim * ca,  " ("+uampl+") \n",
            "Frequency (" + ufreq + ")                 "
            + "Amplitude (" + uampl + ")")
      for i in range(0,npeaks):
          print(fPeak[i] * cf, APeak[i]*ca)
              
      
      
      
  
      
#dirx = "/home/edu/Dropbox/work/aulas/PALESTRAS/ESCOLA_INVERNO_2019/modulo1/"
#filex = dirx + "star01.dat"

#lc1 = LightCurve()
#lc1.readDataFile(filex)
#lc1.plot(ut="hours") 
##lc1.summary() 
#lc1.dft(ufreq='uHz', uampl='mma', limit=True)
#lc1.getPeaks(amin=4., uampl='mma', ufreq='uHz')
##lc1.detectionLimit()




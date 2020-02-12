#!/usr/bin/python3

import numpy as np
import skimage.measure
import cv2 as cv

def gauss(M,center=0.,bwd=1.):
    x=np.fft.fftfreq(M.shape[0])
    y=np.fft.fftfreq(M.shape[1])
    XX,YY = np.meshgrid(x,y)
    RR = np.sqrt(np.power(XX,int(2)) + np.power(YY,int(2)))
    ZZ = np.exp(-np.power(RR/bwd,int(2)))
    return ZZ.T

def cos2(M,center=0.,bwd=1.):
    x=np.fft.fftfreq(M.shape[0])
    y=np.fft.fftfreq(M.shape[1])
    XX,YY = np.meshgrid(x,y)
    ZZ = np.zeros(XX.shape)
    RR = np.sqrt(np.power(XX,int(2)) + np.power(YY,int(2)))
    inds = np.where(RR<bwd)
    ZZ[inds] = 0.5*(1+np.cos(RR[inds]/bwd*np.pi))
    return ZZ.T

def sin2(M,center=0.,bwd=1.):
    x=np.fft.fftfreq(M.shape[0])
    y=np.fft.fftfreq(M.shape[1])
    XX,YY = np.meshgrid(x,y)
    RR = np.sqrt(np.power(XX,int(2)) + np.power(YY,int(2)))
    ZZ = np.zeros(XX.shape)
    inds = np.where(RR<bwd)
    ZZ[inds] = np.power(np.sin(RR[inds]/bwd*np.pi),int(2))
    return ZZ.T

def derivH(M,center=0.,bwd=.1):
    x=np.linspace(-1,1,M.shape[1])
    y=np.linspace(-1,1,M.shape[0])
    XX,YY = np.meshgrid(x,y)
    ZZ = np.power(np.cos(YY/bwd*np.pi/2),int(2)) * (np.abs(YY)/bwd<1.) * np.sin(XX/bwd*np.pi)*np.cos(XX/bwd*np.pi/2.) * (np.abs(XX)/bwd<1)
    ZZ = np.roll(ZZ,ZZ.shape[0]//2,axis=0)
    ZZ = np.roll(ZZ,ZZ.shape[1]//2,axis=1)
    return ZZ

def derivV(M,center=0.,bwd=.1):
    x=np.linspace(-1,1,M.shape[1])
    y=np.linspace(-1,1,M.shape[0])
    XX,YY = np.meshgrid(x,y)
    ZZ = np.power(np.cos(XX/bwd*np.pi/2),int(2)) * (np.abs(XX)/bwd<1.) * np.sin(YY/bwd*np.pi)*np.cos(YY/bwd*np.pi/2.) * (np.abs(YY)/bwd<1)
    ZZ = np.roll(ZZ,ZZ.shape[0]//2,axis=0)
    ZZ = np.roll(ZZ,ZZ.shape[1]//2,axis=1)
    return ZZ

def derivTH(M,th=0.,center=0.,bwd=.1):
    x=np.linspace(-1,1,M.shape[1])
    y=np.linspace(-1,1,M.shape[0])
    XX,YY = np.meshgrid(x,y)
    XXP = XX * np.cos(th) - YY * np.sin(th)
    YYP = XX * np.sin(th) + YY * np.cos(th)
    ZZP = np.power(np.cos(XXP/bwd*np.pi/2),int(2)) * (np.abs(XXP)/bwd<1.) * np.sin(YYP/bwd*np.pi)*np.cos(YYP/bwd*np.pi/2.) * (np.abs(YYP)/bwd<1)
    ZZP = np.roll(ZZP,ZZP.shape[0]//2,axis=0)
    ZZP = np.roll(ZZP,ZZP.shape[1]//2,axis=1)
    return ZZP

def corkscrewTH(M,th=0.,center=0.,bwd=.1):
    x=np.linspace(-1,1,M.shape[1])
    y=np.linspace(-1,1,M.shape[0])
    XX,YY = np.meshgrid(x,y)
    XXP = XX * np.cos(th) - YY * np.sin(th)
    YYP = XX * np.sin(th) + YY * np.cos(th)
    RRP = np.sqrt(np.power(XXP,int(2)) + np.power(YYP,int(2)))
    THP = np.arctan(YYP/XXP)
    ZZP = np.power(np.sin(RRP/bwd*np.pi),int(2)) * (np.abs(RRP)/bwd<1.) * THP
    ZZP = np.roll(ZZP,ZZP.shape[0]//2,axis=0)
    ZZP = np.roll(ZZP,ZZP.shape[1]//2,axis=1)
    return ZZP

def uintBDnorm(x,bd=8,border=2):
    result = np.zeros(x.shape)
    if len(x.shape)>2:
        for i in range(x.shape[2]):
            r = np.copy(x[:,:,i])
            mn = np.min(r[border:-border,border:-border])
            r -= mn
            mx = np.max(r[border:-border,border:-border])
            r *= int(np.power(2,int(bd))-1)//mx
            result[:,:,i] = np.clip(r,0,np.power(int(2),int(bd))-1)
    else:
        r = np.copy(x)
        mn = np.min(r[border:-border,border:-border])
        r -= mn
        mx = np.max(r[border:-border,border:-border])
        r *= int(np.power(2,bd)-1)//mx
        result[:,:] = np.clip(r,0,np.power(int(2),int(bd))-1)
    return result

def uint8norm(x,border=2):
    result = np.zeros(x.shape)
    if len(x.shape)>2:
        for i in range(x.shape[2]):
            r = np.copy(x[:,:,i]).astype(float)
            mn = np.min(r[border:-border,border:-border])
            r -= mn
            mx = np.max(r[border:-border,border:-border])
            r *= 255./mx
            result[:,:,i] = np.clip(r,0,255)
    else:
        r = np.copy(x)
        mn = np.min(r[border:-border,border:-border])
        r -= mn
        mx = np.max(r[border:-border,border:-border])
        r *= 255./mx
        result[:,:] = np.clip(r,0,255)
    return result

def uint8normSign(x,s=1,border=2):
    result = np.zeros(x.shape)
    if len(x.shape)>2:
        for i in range(x.shape[2]):
            r = np.copy(s*x[:,:,i]).astype(float)
            mx = np.max(r[border:-border,border:-border])
            r *= 255./mx
            result[:,:,i] = np.clip(r,0,255)
    else:
        r = np.copy(s*x)
        mx = np.max(r[border:-border,border:-border])
        r *= 255./mx
        result[:,:] = np.clip(r,0,255)
    return result

def loadfile(name):
    data = np.loadtxt(name,skiprows=300)
    im = np.int16(data[:-600,:])     # convert to signed 16 bit integer to allow overflow
    return im
    #return np.copy(im[100:350,800:1200])
    #return np.copy(im[0:550,400:1600])

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_uint8(x):
    return uint8normSign( 1/(1 + np.exp(-x.astype(float))), 1 )

refPt = []
def getclick(event, x, y, flags, param):
    global refPt
    if event == cv.EVENT_LBUTTONDOWN:
        refPt.append([x,y])
    return

def getpearls(outpng,fname):
    #cv.namedWindow("Pearls")
    cv.setMouseCallback("Pearls", getclick)
    while True:
        cv.imshow("Pearls",outpng[:,:,:3].astype(np.uint8))
        key = cv.waitKey(1) & 0xFF
        if key == ord("q"):
            #cv.destroyWindow("Pearls")
            break
    np.savetxt(fname,np.asarray(refPt),fmt='%i')
    return refPt

def getnonpearls(outpng,fname):
    #cv.namedWindow("Non-pearls")
    cv.setMouseCallback("Non-pearls", getclick)
    while True:
        cv.imshow("Non-pearls",outpng[:,:,:3].astype(np.uint8))
        key = cv.waitKey(1) & 0xFF
        if key == ord("q"):
            #cv.destroyWindow("Non-pearls")
            break
    np.savetxt(fname,np.asarray(refPt),fmt='%i')
    return refPt

def maxpool(inmat,kern=2):
        out=np.zeros((inmat.shape[0]//kern,inmat.shape[1]//kern,inmat.shape[2]))
        (sz0,sz1,sz2) = out.shape
        for r in range(sz0):
            for c in range(sz1):
                for l in range(sz2):
                    out[r,c,l] = np.max(inmat[kern*r:kern*(r+1),kern*c:kern*(c+1),l])
        return out



bandwidth = 0.2
nangles = 16
captureTruths = False

def main():
    global refPt,i,ddir,bandwidth,nangles,captureTruths
    angles = np.linspace(-np.pi,np.pi,nangles,endpoint=False)

    filename = "%s/frames_1.mat"%(ddir)
    img = loadfile(filename)


    XXindmap,YYindmap = np.meshgrid(np.arange(img.shape[1]),np.arange(img.shape[0]))
    grad = np.zeros((img.shape[0],img.shape[1],angles.shape[0]),dtype=float)
    GRAD = np.zeros((img.shape[0],img.shape[1],angles.shape[0]),dtype=complex)
    for t in range(nangles):
        grad[:,:,t] = derivTH(img,angles[t],center=0,bwd=.0125)
        GRAD[:,:,t] = np.fft.fft2(grad[:,:,t])

    #tempout = uint8norm(grad[:,:,0]).astype(np.uint8)
    COS2 = cos2(img,0,bwd=bandwidth)
    (sz0,sz1) = COS2.shape
    SIN2 = sin2(img,0,bwd=bandwidth)
    out = np.zeros((sz0,sz1,3),dtype=np.uint8)
    out[:,:,0] = (uint8norm(np.roll(np.roll(np.abs(COS2),sz0//2,axis=0),sz1//2,axis=1))).astype(np.uint8)
    out[:,:,2] = (uint8norm(np.roll(np.roll(np.abs(SIN2),sz0//2,axis=0),sz1//2,axis=1))).astype(np.uint8) 
    cv.imwrite('%s/newoutput/COS2SIN2.png'%(ddir),out)
    '''
    cv.imwrite('%s/newoutput/COS2out.png'%(ddir), (uint8norm(np.roll(np.roll(np.abs(COS2),COS2.shape[0]//2,axis=0),COS2.shape[1],axis=1))).astype(np.uint8)) 
    cv.imwrite('%s/newoutput/SIN2out.png'%(ddir), (uint8norm(np.roll(np.roll(np.abs(SIN2),SIN2.shape[0]//2,axis=0),SIN2.shape[1]//2,axis=1))).astype(np.uint8)) 
    '''
    cos2out = np.fft.ifft2(COS2)
    sin2out = np.fft.ifft2(SIN2)
    out[:,:,0] = (uint8norm(np.roll(np.roll(cos2out.real,sz0//2,axis=0),sz1//2,axis=1))).astype(np.uint8)
    out[:,:,2] = (uint8norm(np.roll(np.roll(sin2out.real,sz0//2,axis=0),sz1//2,axis=1))).astype(np.uint8) 
    frac = int(16)
    cv.imwrite('%s/newoutput/cos2sin2.png'%(ddir),out[(frac//2-1)*sz0//frac:(frac//2+1)*sz0//frac,(frac//2-1)*sz1//frac:(frac//2+1)*sz1//frac,:])
    '''
    cv.imwrite('%s/newoutput/cos2out.png'%(ddir), (uint8norm(np.roll(np.roll(cos2out.real,sz0//2,axis=0),sz1//2,axis=1))).astype(np.uint8)) 
    cv.imwrite('%s/newoutput/sin2out.png'%(ddir), (uint8norm(np.roll(np.roll(sin2out.real,sz0//2,axis=0),sz1//2,axis=1))).astype(np.uint8)) 
    '''
    

    startimage = 20
    stopimage = 92
    for i in range(startimage,stopimage,1):
        filename = "%s/frames_%i.mat"%(ddir,i)
        outname = "%s/out0_%i.dat"%(ddir,i)
        img = loadfile(filename)
        img = uintBDnorm(cv.medianBlur(img,3),12)
        imgcopy = np.copy(img)
        cv.imwrite('%s/newoutput/orig_img%03i.png'%(ddir,i), uint8norm(imgcopy).astype(np.uint8)) 

        del imgcopy

        ## cos2 and sin2 blurs
        IMG = np.fft.fft2(img.astype(float))
        OUT1 = IMG * COS2
        OUT2 = IMG * SIN2
        out1 = uint8norm(np.fft.ifft2( OUT1 ).real,border=4)
        out2 = np.fft.ifft2( OUT2 ).real
        out2 = uint8norm(out2,border=4)


        outpng = np.zeros((img.shape[0], img.shape[1], 9),dtype=int)
        ## stddev blur
        meanmat = np.zeros(img.shape,dtype=float)
        stdmat = np.zeros(img.shape,dtype=float)
        meanmat = np.fft.ifft2(gauss(img,0,.00625)*IMG).real 
        STDMAT = np.fft.fft2(np.power(img - meanmat,int(2)))
        varmat = np.fft.ifft2(gauss(img,0,.025)*STDMAT).real
        stdmat = uint8norm(np.sqrt(varmat))

        ## directional gradients
        outmat = np.zeros((img.shape[0], img.shape[1], nangles),dtype=float)
        for t in range(angles.shape[0]):
            outmat[:,:,t] = np.fft.ifft2(GRAD[:,:,t]*IMG).real
        directionmat = np.argmax(outmat,axis=2).astype(float)
        scalarmat = np.max(outmat,axis=2)

        del outmat

        outpng[:,:,0] = out2.astype(np.uint8)
        outpng[:,:,1] = uint8norm(scalarmat,border=4).astype(np.uint8)
        outpng[:,:,2] = out1.astype(np.uint8)
        outpng[:,:,4] = uint8normSign(stdmat,1,border=4).astype(np.uint8)
        outpng[:,:,3] = (np.cos(directionmat * 2*np.pi/nangles)*127 + 128).astype(np.uint8)
        outpng[:,:,5] = (np.sin(directionmat * 2*np.pi/nangles)*127 + 128).astype(np.uint8)
        #out=maxpool(outpng[:,:,:3],kern=2)
        out=outpng[100:350,800:1200,:3]
        cv.imwrite('%s/newoutput/out1_img%03i.png'%(ddir,i), out) 

        refPt = []
        if captureTruths:
            fname = '%s/newoutput/frame_%03i.pearls'%(ddir,i)
            cv.namedWindow("Pearls")
            pearlcoords = getpearls(outpng,fname)
            refPt = []
            fname = '%s/newoutput/frame_%03i.nonpearls'%(ddir,i)
            cv.namedWindow("Non-pearls")
            nonpearlcoords = getnonpearls(outpng,fname)

        out=outpng[100:350,800:1200,3:6]
        cv.imwrite('%s/newoutput/out2_img%03i.png'%(ddir,i), out) 

        '''
            Encoding for the labels (ground truth) into an out3 png 8 bit file.
        '''
        truth = outpng[:,:,6]
        radii = outpng[:,:,7]
        directions = outpng[:,:,8]
        truth += 127

        radii_lim = 20.
        if captureTruths:
            for (x,y) in pearlcoords:
                inds = np.where(np.power(XXindmap-x,int(2))+np.power(YYindmap-y,int(2)) < radii_lim**2)
                radii[inds] = (np.sqrt(np.power(XXindmap[inds]-x,int(2))+np.power(YYindmap[inds]-y,int(2)))/radii_lim * 256).astype(int)
                directions[inds] = (np.angle((XXindmap[inds]-x) + 1j*(YYindmap[inds]-y))/2./np.pi * 128 + 127).astype(int)
                truth[inds] = 255

            for (x,y) in nonpearlcoords:
                inds = np.where(np.power(XXindmap-x,int(2))+np.power(YYindmap-y,int(2)) < radii_lim**2)
                radii[inds] = (np.sqrt(np.power(XXindmap[inds]-x,int(2))+np.power(YYindmap[inds]-y,int(2)))/radii_lim * 256).astype(int)
                directions[inds] = (np.angle((XXindmap[inds]-x) + 1j * (YYindmap[inds]-y))/2./np.pi * 128 + 127).astype(int)
                truth[inds] = 0 

        out=outpng[100:350,800:1200,6:9]
        cv.imwrite('%s/newoutput/out3_img%03i.png'%(ddir,i), out) 

        if captureTruths:
            cv.namedWindow("Truth")
            tempout = np.zeros((outpng.shape[0],outpng.shape[1],4),dtype=np.uint8)
            tempout[:,:,0] = outpng[:,:,0]
            tempout[:,:,1] = outpng[:,:,1]
            tempout[:,:,2] = outpng[:,:,2]
            tempout[:,:,3] = truth
            cv.imshow("Truth",tempout.astype(np.uint8))
            cv.waitKey(0)
            cv.destroyAllWindows()

            outpng = outpng.reshape((outpng.shape[0]*outpng.shape[1],outpng.shape[2]))
            pearlrows = [row for row in outpng.tolist() if row[6] > 200]
            nonpearlrows = [row for row in outpng.tolist() if row[6] < 100]
            np.savetxt('%s/newoutput/allpixels_img%03i.dat'%(ddir,i),outpng,fmt='%i')
            np.savetxt('%s/newoutput/pearls_img%03i.dat'%(ddir,i),np.array(pearlrows),fmt='%i')
            np.savetxt('%s/newoutput/nonpearls_img%03i.dat'%(ddir,i),np.array(nonpearlrows),fmt='%i')
            print('finished printing pear features')


        print('finished image %i'%(i))

    return

if __name__ == "__main__":
    ddir = "./DataSet2"
    bandwidth =0.4
    nangles = 16
    main()

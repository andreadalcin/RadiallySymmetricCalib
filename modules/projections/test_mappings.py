import unittest
from mappings import * 
import numpy as np


class eqr2eqr(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.H = 11
        self.W = 22

    def test_eqr2sphere(self):
        H = self.H
        W = self.W
        
        EqrDes = Equirectangular_Description(width=W, height=H)

        # sample output space
        i,j = np.meshgrid(np.arange(0,int(H)),np.arange(0,int(W)),indexing="ij")

        x, y, z,_ = EqrDes.image2world([i,j])
        lats = np.arctan2(z,np.sqrt(x**2+y**2))
        longs = np.arctan2(y,x)
        
        self.assertTrue(np.allclose(longs[:,0],-np.pi))
        self.assertTrue(np.allclose(longs[:,-1],np.pi))

        self.assertTrue(np.allclose(lats[0,:],np.pi/2))
        self.assertTrue(np.allclose(lats[-1,:],-np.pi/2))

        i = H//2
        j = 0
        self.assertTrue(np.allclose([x[i,j],y[i,j],z[i,j]],[-1,0,0]))
        self.assertTrue(np.allclose([lats[i,j],longs[i,j]],[0,-np.pi]))

        i = np.array([0, H//2, H-1])
        j = np.array([0, 0, W-1])
        x, y, z, _ = EqrDes.image2world([i,j])
        lats = np.arctan2(z,np.sqrt(x**2+y**2))
        longs = np.arctan2(y,x)
        self.assertTrue(np.allclose(lats, [np.pi/2, 0, -np.pi/2]))
        self.assertTrue(np.allclose(longs, [-np.pi, -np.pi, np.pi]))

    def test_sphere2eqr(self):
        H = self.H
        W = self.W

        EqrDes = Equirectangular_Description(width=W, height=H)
        
        i = np.array([0, H//2, H-1])
        j = np.array([0, 0, W-1])
        x, y, z,_ = EqrDes.image2world([i,j])
        u,v,_= EqrDes.world2image([x,y,z])
        self.assertTrue(np.allclose([u,v],[j,i]))

    def test_mapRotation(self):
        H = self.H
        W = self.W
        i,j = np.meshgrid(np.arange(0,int(H)),np.arange(0,int(W)),indexing="ij")

        img_des = Equirectangular_Description(width=W, height=H, extrinsic_rot=[180,0,0])
        out_des = Equirectangular_Description(width=W, height=H, extrinsic_rot=[0,0,0])
        out_x, out_y, _ = map_points(points=[i,j], des_list=[ img_des, out_des])
        self.assertTrue(np.allclose(out_y, i[::-1]))
        self.assertTrue(np.allclose(out_x[H//2,:], j[H//2,:][::-1]))

        img_des = Equirectangular_Description(width=W, height=H, extrinsic_rot=[0,0,0])
        out_des = Equirectangular_Description(width=W, height=H, extrinsic_rot=[180,0,0])
        out_x, out_y, _ = map_points(points=[i,j], des_list=[ img_des, out_des])

        self.assertTrue(np.allclose(out_y, i[::-1]))
        self.assertTrue(np.allclose(out_x[H//2,:], j[H//2,:][::-1]))

        img_des = Equirectangular_Description(width=W, height=H, extrinsic_rot=[90,0,0])
        out_des = Equirectangular_Description(width=W, height=H, extrinsic_rot=[90,0,0])
        out_x, out_y, _ = map_points(points=[i,j], des_list=[ img_des, out_des])


        self.assertTrue(np.allclose(out_y, i))
        self.assertTrue(np.allclose(out_x[H//2,:], j[H//2,:]))


if __name__ == '__main__':
    unittest.main()
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

class visualization(object):
    def __init__(self, width, height):
        self.K = np.array([[height//2,0,width//2],
                           [0,height//2,height//2],
                           [0,0,1]])
        self.extrinsic = np.array([[1,0,0,0],
                                   [0,1,0,-1],
                                   [0,0,1,-8]])
        self.projection = self.K@self.extrinsic
        self.isOpened = True
        self.keyboard = [0,0,0,0,0,0]
        self.old_rotation = np.array([0,0,0])
        self.old_translation = np.array([0,-1,-8])
        self.coeffs_rotation = np.pi/200
        self.coeffs_translation = 1/100
        self.blank = np.ones((height, width, 3), dtype=np.uint8)*255
        self.windows = cv2.namedWindow("img", cv2.WINDOW_FREERATIO)
        cv2.setMouseCallback('img',self.callback)
        self.polyhedre_draw_order = [[0,2],[0,3],[0,4],[0,5],[1,2],[1,3],[1,4],[1,5]]
        self.axes_correspondance = {"x": [0,0,-90],
                                    "-x": [0,0,90],
                                    "y":[0,0,0],
                                    "-y":[0,0,180],
                                    "z":[90,0,0],
                                    "-z":[-90,0,0]}
        self.revoluteAxesCorrespondance= {"x": [0, True],
                                          "-x": [0,False],
                                          "y": [1,True],
                                          "-y": [1,False],
                                          "z": [2,True],
                                          "-z": [2,False]}

    def defineArm(self, nb_noeud, lengths=[], widths=[], types=[], limits=[], axes=[], axes_rot=[], colors=[]):
        assert len(lengths) == nb_noeud, "Il manque des longueurs pour definir le robot"
        assert len(widths) == nb_noeud, "Il manque des largeurs pour definir le robot"
        assert len(colors) == nb_noeud, "Il manque des couleurs pour definir le robot"
        self.nb_node = nb_noeud
        self.lenghts = lengths
        self.widths = widths
        self.types = types
        self.limits = limits
        self.axes = axes
        self.axes_rot = axes_rot
        self.colors = colors

    def grepRtMat(self, params):
        Rts = {}
        for i in range(self.nb_node):
            if self.types[i] == "revolute":
                if i == 0:
                    lenght = np.array([0,0,0])
                else:
                    if "x" in self.axes[i-1]:
                        if self.types[i-1] == "prismatic":
                            lenght = np.array([self.lenghts[i-1]+params[i-1],0,0])
                        else:
                            lenght = np.array([self.lenghts[i-1],0,0])
                    elif "y" in self.axes[i-1]:
                        if self.types[i-1] == "prismatic":
                            lenght = np.array([0,self.lenghts[i-1]+params[i-1],0])
                        else:
                            lenght = np.array([0,self.lenghts[i-1],0])
                    elif "z" in self.axes[i-1]:
                        if self.types[i-1] == "prismatic":
                            lenght = np.array([0,0,self.lenghts[i-1]+params[i-1]])
                        else:
                            lenght = np.array([0,0,self.lenghts[i-1]])
                correspondance = self.revoluteAxesCorrespondance[self.axes_rot[i]]
                if correspondance[1]:
                    rot_local = [0,0,0]
                    rot_local[correspondance[0]] += params[i]
                else:
                    rot_local = [0,0,0]
                    rot_local[correspondance[0]] += -params[i]
                rmat = R.from_euler("xyz", rot_local, degrees=True).as_matrix()
            else:
                if i == 0:
                    lenght = np.array([0,0,0])
                else:
                    if "x" in self.axes[i-1]:
                        if self.types[i-1] == "prismatic":
                            lenght = np.array([self.lenghts[i-1]+params[i-1],0,0])
                        else:
                            lenght = np.array([self.lenghts[i-1],0,0])
                    elif "y" in self.axes[i-1]:
                        if self.types[i-1] == "prismatic":
                            lenght = np.array([0,self.lenghts[i-1]+params[i-1],0])
                        else:
                            lenght = np.array([0,self.lenghts[i-1],0])
                    elif "z" in self.axes[i-1]:
                        if self.types[i-1] == "prismatic":
                            lenght = np.array([0,0,self.lenghts[i-1]+params[i-1]])
                        else:
                            lenght = np.array([0,0,self.lenghts[i-1]])
                rmat = np.eye(3)
            Rts[i] = np.eye(4)
            Rts[i][:3,:3] = rmat
            Rts[i][:3,3] = lenght
            if i > 0:
                Rts[i] = Rts[i-1]@Rts[i]
        return Rts

    def DrawArm(self, img, params):
        Rts = self.grepRtMat(params)
        for i in range(self.nb_node):
            if self.types[i] == "revolute":
                img = self.drawRevolute(img, Rts[i], self.lenghts[i], self.widths[i], self.axes[i], self.colors[i])
            else:
                img = self.drawPrismatic(img, Rts[i], self.lenghts[i], params[i], self.widths[i], self.axes[i], self.colors[i])
        # print(Rts)
        return img

    def getPolyhedre(self, l, w, axe):
        poly = np.expand_dims(np.array([[0,0,0],
                         [0,l,0],
                         [w,l/4,w],
                         [w,l/4,-w],
                         [-w,l/4,w],
                         [-w,l/4,-w]]), axis=2)
        Rt = np.eye(4)
        Rt[:3,:3] = R.from_euler("xyz",self.axes_correspondance[axe], degrees=True).as_matrix()
        poly = self.addOnes(poly)
        poly = Rt@poly
        return poly

    def drawPolyhedre(self, img, Rt, poly, color):
        poly = Rt@poly
        poly[:,:3] /= poly[:,3:]
        poly = poly[:,:3]
        poly = np.squeeze(poly, axis=2)
        for i in range(len(self.polyhedre_draw_order)):
            self.Line(poly[self.polyhedre_draw_order[i][0]], poly[self.polyhedre_draw_order[i][1]], color, img, 1)
        return img

    def drawPrismatic(self, img, Rt, lenght, command_lenght, width, axe, color):
        poly = self.getPolyhedre(lenght+command_lenght, width, axe)
        img = self.drawPolyhedre(img, Rt, poly, color)
        return img

    def drawRevolute(self, img, Rt, lenght, width, axe, color):
        poly = self.getPolyhedre(lenght, width, axe)
        img = self.drawPolyhedre(img, Rt, poly, color)
        return img

    @property
    def gauche(self):
        return True if self.keyboard[0] == 1 else False
    
    @property
    def droite(self):
        return True if self.keyboard[1] == 1 else False

    @property
    def haut(self):
        return True if self.keyboard[2] == 1 else False

    @property
    def bas(self):
        return True if self.keyboard[3] == 1 else False
    
    @property
    def shift(self):
        return True if self.keyboard[4] == 1 else False

    @property
    def ctrl(self):
        return True if self.keyboard[5] == 1 else False

    def newProjectionMatrix(self, rot, trans, offsetTrans, trans_mod=False):
        self.extrinsic = np.zeros((3,4))
        rmat = R.from_euler("xyz", rot).as_matrix()
        self.extrinsic[:3,:3] = rmat
        if trans_mod:
            self.extrinsic[:3,3] = trans + (rmat@offsetTrans).T
        else:
            self.extrinsic[:3,3] = trans
        self.projection = self.K@self.extrinsic

    def callback(self, event, x, y, flags, param):
        current = [x,y]
        if flags!= 0:
            if flags == 1:
                x_offset = current[0]-self.click_droit_init[0]
                y_offset = current[1]-self.click_droit_init[1]
                rot_z = self.coeffs_rotation*x_offset
                rot_x = self.coeffs_rotation*y_offset
                self.newProjectionMatrix([self.old_rotation[0]+rot_x, self.old_rotation[1]+rot_z, self.old_rotation[2]], self.old_translation, np.array([[0],[0],[0]]))
            elif flags == 2:
                x_offset = current[0]-self.click_gauche_init[0]
                y_offset = current[1]-self.click_gauche_init[1]
                tran_x = -self.coeffs_translation*x_offset
                tran_y = -self.coeffs_translation*y_offset
                self.newProjectionMatrix(self.old_rotation, self.old_translation, np.array([[tran_x], [tran_y], [0]]), True)
            elif flags == 4:
                x_offset = current[0]-self.click_middle_init[0]
                y_offset = current[1]-self.click_middle_init[1]
                tran_x = -self.coeffs_translation*x_offset
                tran_y = -self.coeffs_translation*y_offset
                self.newProjectionMatrix(self.old_rotation, self.old_translation, np.array([[0], [0], [tran_y]]), True)

        if event != 0:
            if event == 1:
                self.old_rotation = R.from_matrix(self.extrinsic[:3,:3]).as_euler("xyz")
                self.old_translation = self.extrinsic[:3,3]
                self.click_droit_init = current
            elif event == 2:
                self.old_rotation = R.from_matrix(self.extrinsic[:3,:3]).as_euler("xyz")
                self.old_translation = self.extrinsic[:3,3]
                self.click_gauche_init = current
            elif event == 3:
                self.old_rotation = R.from_matrix(self.extrinsic[:3,:3]).as_euler("xyz")
                self.old_translation = self.extrinsic[:3,3]
                self.click_middle_init = current
        pass

    def addOnes(self, pts):
        if pts.shape[0] == 2 and len(pts.shape) <= 2:
            if len(pts.shape) == 1:
                pts = np.expand_dims(pts, axis=1)
            pts = np.vstack([pts,np.ones((1,1))])
        elif pts.shape[0] == 3 and len(pts.shape) <= 2:
            if len(pts.shape) == 1:
                pts = np.expand_dims(pts, axis=1)
            pts = np.vstack([pts,np.ones((1,1))])
        else:
            if len(pts.shape) == 3 and pts.shape[1] == 3:
                pts = np.hstack([pts, np.ones((pts.shape[0],1,1))])
            elif len(pts.shape) == 3 and pts.shape[1] == 2:
                pts = np.hstack([pts, np.ones((pts.shape[0],1,1))])
        return pts
    
    def normalize(self, pts):
        if pts.shape[0] == 3:
            pts[:2,:] /= pts[2:,:]
            pts = np.squeeze(pts[:2],axis=1)
        elif pts.shape[0] == 4:
            pts[:3,:] /= pts[3:,:]
            pts = np.squeeze(pts[:3,],axis=1)
        return pts

    def project(self, pts):
        assert pts.shape[0] == 3, f"bad dimension, the shape must be (3) or (3,1), actual points shape: {pts.shape}"
        pts = self.addOnes(pts)
        pts = self.projection@pts
        pts = self.normalize(pts)
        return pts

    def Line(self, pts1, pts2, color=(0,255,0), img=None, thickness=2):
        if img is None:
            img = self.blank.copy()
        pts1 = self.project(pts1)
        pts2 = self.project(pts2)
        cv2.line(img, pts1.astype(int), pts2.astype(int), color, thickness)
        return img

    def drawFrame(self, img):
        origine = np.array([0,0,0])
        x = np.array([1,0,0])
        y = np.array([0,1,0])
        z = np.array([0,0,1])
        img = self.Line(origine, x, (0,0,255), img)
        img = self.Line(origine, y, (0,255,0), img)
        img = self.Line(origine, z, (255,0,0), img)
        return img

    def drawgrid(self, width=10, height=10, step=10, color=(78,78,78), img=None, frame=True):
        if img is None:
            img = self.blank.copy()
        step = step - (step%2)
        width = width - (width%2)
        height = height - (height%2)
        for i in range(step+1):
            offset_x = i-(step//2)
            x1 = -width/2
            x2 = width/2
            y1 = -height/2
            y2 = height/2
            pts1_v = np.array([offset_x, 0, y1])
            pts2_v = np.array([offset_x, 0, y2])
            self.Line(pts1_v, pts2_v, color, img, 1)
            for j in range(step+1):
                offset_y = j-(step//2)
                pts1_h = np.array([x1, 0, offset_y])
                pts2_h = np.array([x2, 0, offset_y])
                self.Line(pts1_h, pts2_h, color, img, 1)
        if frame:
            img = self.drawFrame(img)
        return img

    def draw(self, img):
        cv2.imshow("img", img)
        key = cv2.waitKey(1)
        self.keyboard = [0,0,0,0,0,0]
        # if key != -1:
        #     print(key) 
        if key == ord("q") or key == 27:
            self.isOpened = False
        elif key == 81: # gauche
            self.keyboard[0] = 1
        elif key == 83: # droite
            self.keyboard[1] = 1
        elif key == 82: # haut
            self.keyboard[2] = 1
        elif key == 84: # bas
            self.keyboard[3] = 1
        elif key == 225: # shift
            self.keyboard[4] = 1
        elif key == 227: # ctrl
            self.keyboard[5] = 1

if __name__ == "__main__":
    viz = visualization(1027, 768)
    pts1 = np.array([0,0,0])
    pts2 = np.array([0,1,0])
    
    viz.defineArm(3, 
                  lengths=[1,1,1],
                  widths=[0.2,0.15,0.1],
                  types=["revolute", "prismatic", "revolute"],
                  limits=[[-90,90],
                          [0.1,1.1],
                          [-90,90]],
                  axes=["y","x","x"],
                  axes_rot=["y","x","z"],
                  colors=[(255,0,0),(0,255,0),(0,0,255)])

    theta1, delta2, theta3 = 0,0,0
    while viz.isOpened:
        if viz.gauche:
            theta1 += 10
            print(theta1)
        elif viz.droite:
            theta1 -= 10
            print(theta1)
        elif viz.haut:
            delta2 += 0.1
        elif viz.bas:
            delta2 -= 0.1
        elif viz.shift:
            theta3 += 10
            print(theta3)
        elif viz.ctrl:
            theta3 -= 10
            print(theta3)
        img = viz.drawgrid()
        img = viz.DrawArm(img, [theta1,delta2,theta3])
        viz.draw(img)
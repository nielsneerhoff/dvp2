/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package tudelft.cgv.volume;

import java.io.File;
import java.io.IOException;

/**
 *
 * @author michel modified by Anna
 */

//////////////////////////////////////////////////////////////////////
///////////////// CONTAINS FUNCTIONS TO BE IMPLEMENTED ///////////////
//////////////////////////////////////////////////////////////////////

public class Volume {
    

    //Do NOT modify these attributes
    private int dimX, dimY, dimZ;
    private short[] data;
    private int[] histogram;

    // Do NOT modify this function
    // This function returns the nearest neighbour given a position in the volume given by coord.
    public short getVoxelNN(double[] coord) {
        if (coord[0] < 0 || coord[0] > (dimX-1) || coord[1] < 0 || coord[1] > (dimY-1)
                || coord[2] < 0 || coord[2] > (dimZ-1)) {
            return 0;
        }
        /* notice that in this framework we assume that the distance between neighbouring voxels is 1 in all directions*/
        int x = (int) Math.round(coord[0]); 
        int y = (int) Math.round(coord[1]);
        int z = (int) Math.round(coord[2]);
    
        return getVoxel(x, y, z);
    }
        

    //Do NOT modify this function
    //This function linearly interpolates the value g0 and g1 given the factor (t) 
    //the result is returned. It is used for the tri-linearly interpolation the values 
    private float interpolate(float g0, float g1, float factor) {
        float result = (1 - factor)*g0 + factor*g1;
        return result; 
    }
             
    //Do NOT modify this function
    // This function returns the trilinear interpolated value of the position given by  position coord.
    public float getVoxelLinearInterpolate(double[] coord) {
        if (coord[0] < 0 || coord[0] > (dimX-2) || coord[1] < 0 || coord[1] > (dimY-2)
                || coord[2] < 0 || coord[2] > (dimZ-2)) {
            return 0;
        }
        /* notice that in this framework we assume that the distance between neighbouring voxels is 1 in all directions*/
        int x = (int) Math.floor(coord[0]); 
        int y = (int) Math.floor(coord[1]);
        int z = (int) Math.floor(coord[2]);
        
        float fac_x = (float) coord[0] - x;
        float fac_y = (float) coord[1] - y;
        float fac_z = (float) coord[2] - z;

        float t0 = interpolate(getVoxel(x, y, z), getVoxel(x+1, y, z), fac_x);
        float t1 = interpolate(getVoxel(x, y+1, z), getVoxel(x+1, y+1, z), fac_x);
        float t2 = interpolate(getVoxel(x, y, z+1), getVoxel(x+1, y, z+1), fac_x);
        float t3 = interpolate(getVoxel(x, y+1, z+1), getVoxel(x+1, y+1, z+1), fac_x);
        float t4 = interpolate(t0, t1, fac_y);
        float t5 = interpolate(t2, t3, fac_y);
        float t6 = interpolate(t4, t5, fac_z);
        
        return t6; 
    }


    float a = -0.75f; // global variable that defines the value of a used in cubic interpolation.
    // you need to chose the right value
        
    //////////////////////////////////////////////////////////////////////
    ///////////////// FUNCTION TO BE IMPLEMENTED /////////////////////////
    ////////////////////////////////////////////////////////////////////// 
        
    // Function that computes the weights for one of the 4 samples involved in the 1D interpolation. 
    public float weight (float t, Boolean one_two_sample)
    {
         float absoluteT = Math.abs(t);
         float absoluteTSquared = (float) Math.pow(absoluteT,2);
         float absoluteTCubed = (float) Math.pow(absoluteT,3);

         // Implemented formula on the page 26 of the slides.
         if(0 <= absoluteT && absoluteT <1){
            return (a + 2) * absoluteTCubed - (a + 3) * absoluteTSquared + 1;
        } else if(1 <= absoluteT && absoluteT < 2) {
            return a * absoluteTCubed - 5 * a * absoluteTSquared + 8 * a * absoluteT - 4 * a;
        } else {
            return 0f;
        }
   }
    
    //////////////////////////////////////////////////////////////////////
    ///////////////// FUNCTION TO BE IMPLEMENTED /////////////////////////
    ////////////////////////////////////////////////////////////////////// 
    // Function that computes the 1D cubic interpolation. g0,g1,g2,g3 contain the values of the voxels that we want to interpolate
    // factor contains the distance from the value g1 to the position we want to interpolate to.
    // We assume the out of bounce checks have been done earlier

    public float cubicinterpolate (float g0, float g1, float g2, float g3, float factor) {
        
        // We assume that the distance between neighbouring voxels is 1 in all directions,
        // so h(t) takes as argument t = ((x - i delta_x) / delta_x) = factor - i.
        float weight0 = weight(factor + 1, false);
        float weight1 = weight(factor, false);
        float weight2 = weight(factor - 1, false);
        float weight3 = weight(factor - 2, false);
        
        // Take the weighted sum of our 3 voxels.
        float expectedValue = g0 * weight0 + g1 * weight1 + g2 * weight2 + g3 * weight3;
        
        // Clamp the values such that no negative values may be returned.
        // Limit value to 255, otherwise causes ArrayIndexOutOfBoundsException.
        expectedValue = Math.max(0, expectedValue);
        expectedValue = Math.min(255, expectedValue); 

        return expectedValue;
    }
        
    //////////////////////////////////////////////////////////////////////
    ///////////////// FUNCTION TO BE IMPLEMENTED /////////////////////////
    ////////////////////////////////////////////////////////////////////// 
    // 2D cubic interpolation implemented here. We do it for plane XY. Coord contains the position.
    // We assume the out of bounce checks have been done earlier
    public float bicubicinterpolateXY (double[] coord, int z) {
        int x = (int) Math.floor(coord[0]);
        int y = (int) Math.floor(coord[1]);
        
        // Compute factor, the distance from voxel g1 to the position we want to interpolate to.
        float factorX = (float) (coord[0] - x);
        float factorY = (float) (coord[1] - y);
        
        // Interpolate four points along the x-axis.
        float x0 = cubicinterpolate(
                (float) getVoxel(x - 1, y - 1, z), 
                (float) getVoxel(x, y - 1, z), 
                (float) getVoxel(x + 1, y - 1, z), 
                (float) getVoxel(x + 2, y - 1, z), 
                factorX);
        float x1 = cubicinterpolate(
                (float) getVoxel(x - 1, y, z), 
                (float) getVoxel(x, y, z), 
                (float) getVoxel(x + 1, y, z), 
                (float) getVoxel(x + 2, y, z), 
                factorX);
        float x2 = cubicinterpolate(
                (float) getVoxel(x - 1, y + 1, z), 
                (float) getVoxel(x, y + 1, z), 
                (float) getVoxel(x + 1, y + 1, z), 
                (float) getVoxel(x + 2, y + 1, z), 
                factorX);
        float x3 = cubicinterpolate(
                (float) getVoxel(x - 1, y + 2, z), 
                (float) getVoxel(x, y + 2, z), 
                (float) getVoxel(x + 1, y + 2, z), 
                (float) getVoxel(x + 2, y + 2, z), 
                factorX);

        return cubicinterpolate(x0, x1, x2, x3, factorY);
    }
            
    //////////////////////////////////////////////////////////////////////
    ///////////////// FUNCTION TO BE IMPLEMENTED /////////////////////////
    ////////////////////////////////////////////////////////////////////// 
    // 3D cubic interpolation implemented here given a position in the volume given by coord.
    
    public float getVoxelTriCubicInterpolate (double[] coord) {
        if (coord[0] < 1 || coord[0] > (dimX-3) || coord[1] < 1 || coord[1] > (dimY-3)
                || coord[2] < 1 || coord[2] > (dimZ-3)) {
            return 0;
        }
        
        int z = (int) Math.floor(coord[2]);
        
        // Compute factor, the distance from voxel g1 to the position we want to interpolate to.
        float factorZ = (float) (coord[2] - z);

        // Interpolate four points along the z-axis.
        float y0 = bicubicinterpolateXY(coord, z - 1);
        float y1 = bicubicinterpolateXY(coord, z);
        float y2 = bicubicinterpolateXY(coord, z + 1);
        float y3 = bicubicinterpolateXY(coord, z + 2);

        return cubicinterpolate(y0, y1, y2, y3, factorZ);
    }


//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

	
    //Do NOT modify this function
    public Volume(int xd, int yd, int zd) {
        data = new short[xd*yd*zd];
        dimX = xd;
        dimY = yd;
        dimZ = zd;
    }
    //Do NOT modify this function
    public Volume(File file) {
        
        try {
            VolumeIO reader = new VolumeIO(file);
            dimX = reader.getXDim();
            dimY = reader.getYDim();
            dimZ = reader.getZDim();
            data = reader.getData().clone();
            computeHistogram();
        } catch (IOException ex) {
            System.out.println("IO exception");
        }
        
    }
    
    //Do NOT modify this function
    public short getVoxel(int x, int y, int z) {
    	int i = x + dimX*(y + dimY * z);
        return data[i];
    }
    
    //Do NOT modify this function
    public void setVoxel(int x, int y, int z, short value) {
    	int i = x + dimX*(y + dimY * z);
        data[i] = value;
    }
    
	//Do NOT modify this function
    public void setVoxel(int i, short value) {
        data[i] = value;
    }
    
    //Do NOT modify this function
    public short getVoxel(int i) {
        return data[i];
    }
    
	//Do NOT modify this function
    public int getDimX() {
        return dimX;
    }
    
    //Do NOT modify this function
    public int getDimY() {
        return dimY;
    }
    
    //Do NOT modify this function
    public int getDimZ() {
        return dimZ;
    }

    //Do NOT modify this function
    public short getMinimum() {
        short minimum = data[0];
        for (int i=0; i<data.length; i++) {
            minimum = data[i] < minimum ? data[i] : minimum;
        }
        return minimum;
    }
    
    //Do NOT modify this function
    public short getMaximum() {
        short maximum = data[0];
        for (int i=0; i<data.length; i++) {
            maximum = data[i] > maximum ? data[i] : maximum;
        }
        return maximum;
    }
 
    //Do NOT modify this function
    public int[] getHistogram() {
        return histogram;
    }
    
    //Do NOT modify this function
    private void computeHistogram() {
        histogram = new int[getMaximum() + 1];
        for (int i=0; i<data.length; i++) {
            histogram[data[i]]++;
        }
    }
}

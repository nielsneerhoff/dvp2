/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package tudelft.cgv.volume;

/**
 *
 * @author michel and modified by Anna Vilanova 
 * 
 * 
 */

//////////////////////////////////////////////////////////////////////
///////////////// CONTAINS FUNCTIONS TO BE IMPLEMENTED ///////////////
//////////////////////////////////////////////////////////////////////
public class GradientVolume {

	
    
//Do NOT modify this attributes
    private int dimX, dimY, dimZ;
    private VoxelGradient zero = new VoxelGradient();
    VoxelGradient[] data;
    Volume volume;
    double maxmag;
    
//If needed add new attributes here:


    //Do NOT modify this function
    // 
    // Computes the gradient of the volume attribute and save it into the data attribute
    // This is a lengthy computation and is performed only once (have a look at the constructor GradientVolume) 
    //
    private void compute() {

        for (int i=0; i<data.length; i++) {
            data[i] = zero;
        }
       
        for (int z=1; z<dimZ-1; z++) {
            for (int y=1; y<dimY-1; y++) {
                for (int x=1; x<dimX-1; x++) {
                    float gx = (volume.getVoxel(x+1, y, z) - volume.getVoxel(x-1, y, z))/2.0f;
                    float gy = (volume.getVoxel(x, y+1, z) - volume.getVoxel(x, y-1, z))/2.0f;
                    float gz = (volume.getVoxel(x, y, z+1) - volume.getVoxel(x, y, z-1))/2.0f;
                    VoxelGradient grad = new VoxelGradient(gx, gy, gz);
                    setGradient(x, y, z, grad);
                }
            }
        }
        maxmag=calculateMaxGradientMagnitude();
     }
    
    
    
    
    
    
    
    
    //////// NEW

    float a = -0.75f; // global variable that defines the value of a used in cubic interpolation. TODO: Choose right value.
        
    // Function that computes the weights for one of the 4 samples involved in the interpolation. 
    public float weight (float t, Boolean one_two_sample){
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
    
    public float cubicInterpolate (float g0, float g1, float g2, float g3, float factor) {
        
        // We assume that the distance between neighbouring voxels is 1 in all directions,
        // so h(t) takes as argument t = ((x - i delta_x) / delta_x) = factor - i.
        float weight0 = weight(factor + 1, false);
        float weight1 = weight(factor, false);
        float weight2 = weight(factor - 1, false);
        float weight3 = weight(factor - 2, false);
        
        // Take the weighted sum of our 3 voxels.
        float expectedValue = g0 * weight0 + g1 * weight1 + g2 * weight2 + g3 * weight3;
        
        // Clamp the values such that no negative values may be returned.
//        expectedValue = Math.max(0, expectedValue);

        return expectedValue;
    }
    
    // Returns the cubic-interpolated gradient of 4 gradients in 1D.
    public VoxelGradient gradientInterpolate(VoxelGradient g0, VoxelGradient g1, VoxelGradient g2, VoxelGradient g3, float factor) {        
        return new VoxelGradient(
                cubicInterpolate(g0.x, g1.x, g2.x, g3.x, factor), 
                cubicInterpolate(g0.y, g1.y, g2.y, g3.y, factor), 
                cubicInterpolate(g0.z, g1.z, g2.z, g3.z, factor));
    }
    
    // Returns the cubic-interpolated gradient of 4 x 4 gradients (hence 2D).
    public VoxelGradient bicubicinterpolateXY (double[] coord, int z) {
        int x = (int) Math.floor(coord[0]);
        int y = (int) Math.floor(coord[1]);
        
        // Compute factor, the distance from gradient g1 to the position we want to interpolate to.
        float factorX = (float) (coord[0] - x);
        float factorY = (float) (coord[1] - y);
        
        // Interpolate four points along the x-axis.
        VoxelGradient x0 = gradientInterpolate(
                getGradient(x - 1, y - 1, z), 
                getGradient(x, y - 1, z), 
                getGradient(x + 1, y - 1, z), 
                getGradient(x + 2, y - 1, z), 
                factorX);
        VoxelGradient x1 = gradientInterpolate(
                getGradient(x - 1, y, z), 
                getGradient(x, y, z), 
                getGradient(x + 1, y, z), 
                getGradient(x + 2, y, z), 
                factorX);
        VoxelGradient x2 = gradientInterpolate(
                getGradient(x - 1, y + 1, z), 
                getGradient(x, y + 1, z), 
                getGradient(x + 1, y + 1, z), 
                getGradient(x + 2, y + 1, z), 
                factorX);
        VoxelGradient x3 = gradientInterpolate(
                getGradient(x - 1, y + 2, z), 
                getGradient(x, y + 2, z), 
                getGradient(x + 1, y + 2, z), 
                getGradient(x + 2, y + 2, z), 
                factorX);

        return gradientInterpolate(x0, x1, x2, x3, factorY);
    }
    
    // Returns the cubic-interpolated gradient of 4 x 4 gradients (hence 2D).
    public VoxelGradient tricubicInterpolate (double[] coord) {
        if (coord[0] < 1 || coord[0] > (dimX-3) || coord[1] < 1 || coord[1] > (dimY-3)
                || coord[2] < 1 || coord[2] > (dimZ-3)) {
            return new VoxelGradient(0, 0, 0);
        }
        
        int z = (int) Math.floor(coord[2]);
        
        // Compute factor, the distance from voxel g1 to the position we want to interpolate to.
        float factorZ = (float) (coord[2] - z);

        // Interpolate four points along the z-axis.
        VoxelGradient y0 = bicubicinterpolateXY(coord, z - 1);
        VoxelGradient y1 = bicubicinterpolateXY(coord, z);
        VoxelGradient y2 = bicubicinterpolateXY(coord, z + 1);
        VoxelGradient y3 = bicubicinterpolateXY(coord, z + 2);

        return gradientInterpolate(y0, y1, y2, y3, factorZ);
    }
    
//////// NEW
    	
    
    
    
    
    
    
    
    
    
    
    
//////////////////////////////////////////////////////////////////////
///////////////// FUNCTION TO BE IMPLEMENTED /////////////////////////
//////////////////////////////////////////////////////////////////////
//This function linearly interpolates gradient vector g0 and g1 given the factor (t) 
//the resut is given at result. You can use it to tri-linearly interpolate the gradient 
public void linearInterpolate(VoxelGradient g0, VoxelGradient g1, float factor, VoxelGradient result) {
            
        result.x = g0.x * (1 - factor) + g1.x * factor;
        result.y = g0.y * (1 - factor) + g1.y * factor;
        result.z = g0.z * (1 - factor) + g1.z * factor;
        
        result.mag = (float) Math.sqrt(result.x * result.x + result.y * result.y + result.z * result.z);
}
	
//////////////////////////////////////////////////////////////////////
///////////////// FUNCTION TO BE IMPLEMENTED /////////////////////////
//////////////////////////////////////////////////////////////////////
// This function should return linearly interpolated gradient for position coord[].
    public VoxelGradient getGradient(double[] coord) {
        
        if (coord[0] < 0 || coord[0] > (dimX-2) || coord[1] < 0 || coord[1] > (dimY-2)
                || coord[2] < 0 || coord[2] > (dimZ-2)) {
            return zero;
        }
        
        // We assume that the distance between neighbouring voxels is 1 in all directions.
        int x = (int) Math.floor(coord[0]); 
        int y = (int) Math.floor(coord[1]);
        int z = (int) Math.floor(coord[2]);
        
        float factorX = (float) coord[0] - x;
        float factorY = (float) coord[1] - y;
        float factorZ = (float) coord[2] - z;
        
        VoxelGradient t0 = new VoxelGradient(0,0,0);
        VoxelGradient t1 = new VoxelGradient(0,0,0);
        VoxelGradient t2 = new VoxelGradient(0,0,0);
        VoxelGradient t3 = new VoxelGradient(0,0,0);
        VoxelGradient t4 = new VoxelGradient(0,0,0);
        VoxelGradient t5 = new VoxelGradient(0,0,0);
        VoxelGradient t6 = new VoxelGradient(0,0,0);

        linearInterpolate(getGradient(x, y, z), getGradient(x+1, y, z), factorX, t0);
        linearInterpolate(getGradient(x, y+1, z), getGradient(x+1, y+1, z), factorX, t1);
        linearInterpolate(getGradient(x, y, z+1), getGradient(x+1, y, z+1), factorX, t2);
        linearInterpolate(getGradient(x, y+1, z+1), getGradient(x+1, y+1, z+1), factorX, t3);
        linearInterpolate(t0, t1, factorY, t4);
        linearInterpolate(t2, t3, factorY, t5);
        linearInterpolate(t4, t5, factorZ, t6);
        
        return t6; 
    }
    
    
    //Do NOT modify this function
    public VoxelGradient getGradientNN(double[] coord) {
        if (coord[0] < 0 || coord[0] > (dimX-2) || coord[1] < 0 || coord[1] > (dimY-2)
                || coord[2] < 0 || coord[2] > (dimZ-2)) {
            return zero;
        }

        int x = (int) Math.round(coord[0]);
        int y = (int) Math.round(coord[1]);
        int z = (int) Math.round(coord[2]);
        return getGradient(x, y, z);
    }
    
    //Returns the maximum gradient magnitude
    //The data array contains all the gradients, in this function you have to return the maximum magnitude of the vectors in data[] 
    //Do NOT modify this function
    private double calculateMaxGradientMagnitude() {
        if (maxmag >= 0) {
            return maxmag;
        } else {
            double magnitude = data[0].mag;
            for (int i=0; i<data.length; i++) {
                magnitude = data[i].mag > magnitude ? data[i].mag : magnitude;
            }   
            maxmag = magnitude;
            return magnitude;
        }
    }
    
    //Do NOT modify this function
    public double getMaxGradientMagnitude()
    {
        return this.maxmag;
    }
	//////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////
	
	
	//Do NOT modify this function
	public GradientVolume(Volume vol) {
        volume = vol;
        dimX = vol.getDimX();
        dimY = vol.getDimY();
        dimZ = vol.getDimZ();
        data = new VoxelGradient[dimX * dimY * dimZ];
        maxmag = -1.0;
        compute();
    }

	//Do NOT modify this function
	public VoxelGradient getGradient(int x, int y, int z) {
        return data[x + dimX * (y + dimY * z)];
    }

  
  
    //Do NOT modify this function
    public void setGradient(int x, int y, int z, VoxelGradient value) {
        data[x + dimX * (y + dimY * z)] = value;
    }

    //Do NOT modify this function
    public void setVoxel(int i, VoxelGradient value) {
        data[i] = value;
    }
    
    //Do NOT modify this function
    public VoxelGradient getVoxel(int i) {
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

}

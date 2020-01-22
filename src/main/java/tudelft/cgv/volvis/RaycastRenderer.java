// Version for students

package tudelft.cgv.volvis;

import com.jogamp.opengl.GL;
import com.jogamp.opengl.GL2;
import com.jogamp.opengl.util.texture.Texture;
import com.jogamp.opengl.util.texture.awt.AWTTextureIO;
import tudelft.cgv.gui.RaycastRendererPanel;
import tudelft.cgv.gui.TransferFunction2DEditor;
import tudelft.cgv.gui.TransferFunctionEditor;
import java.awt.image.BufferedImage;
import tudelft.cgv.util.TFChangeListener;
import tudelft.cgv.util.VectorMath;
import tudelft.cgv.volume.GradientVolume;
import tudelft.cgv.volume.Volume;
import tudelft.cgv.volume.VoxelGradient;
 
import java.awt.Color;
import java.util.Arrays;


/**
 *
 * @author michel
 *  Edit by AVilanova & Nicola Pezzotti
 * 
 * 
 * Main functions to implement the volume rendering
 */


//////////////////////////////////////////////////////////////////////
///////////////// CONTAINS FUNCTIONS TO BE IMPLEMENTED ///////////////
//////////////////////////////////////////////////////////////////////

public class RaycastRenderer extends Renderer implements TFChangeListener {


// attributes
    
    private Volume volume = null;
    private GradientVolume gradients = null;
    RaycastRendererPanel panel;
    TransferFunction tFunc;
    TransferFunction2D tFunc2D;
    TransferFunctionEditor tfEditor;
    TransferFunction2DEditor tfEditor2D;
    private boolean mipMode = false;
    private boolean slicerMode = true;
    private boolean compositingMode = false;
    private boolean tf2dMode = false;
    private boolean shadingMode = false;
    private boolean silhouetteMode = false;
    private double[] normalLightVector = new double[3];
    private double baseOpacity = 0.1;
    private double sharpness = 0.25;
    private boolean isoMode = false;
    private float iso_value=95; 
    // This is a work around
    private float res_factor = 1.0f;
    private float max_res_factor=0.25f;
    private TFColor isoColor; 

    
    //////////////////////////////////////////////////////////////////////
    ///////////////// FUNCTION TO BE MODIFIED    /////////////////////////
    ////////////////////////////////////////////////////////////////////// 
    // Function that updates the "image" attribute (result of renderings)
    // using the slicing technique. 
    
    public void slicer(double[] viewMatrix) {
	
        // we start by clearing the image
        resetImage();

        // vector uVec and vVec define the view plane, 
        // perpendicular to the view vector viewVec which is going from the view point towards the object
        // uVec contains the up vector of the camera in world coordinates (image vertical)
        // vVec contains the horizontal vector in world coordinates (image horizontal)
        double[] viewVec = new double[3];
        double[] uVec = new double[3];
        double[] vVec = new double[3];
        getViewPlaneVectors(viewMatrix,viewVec,uVec,vVec);

        // The result of the visualization is saved in an image(texture)
        // we update the vector according to the resolution factor
        // If the resolution is 0.25 we will sample 4 times more points. 
        for(int k=0;k<3;k++)
        {
            uVec[k]=res_factor*uVec[k];
            vVec[k]=res_factor*vVec[k];
        }

        // compute the volume center
        double[] volumeCenter = new double[3];
        computeVolumeCenter(volumeCenter);

        // Here will be stored the 3D coordinates of every pixel in the plane 
        double[] pixelCoord = new double[3];

        // We get the size of the image/texture we will be puting the result of the 
        // volume rendering operation.
        int imageW=image.getWidth();
        int imageH=image.getHeight();

        int[] imageCenter = new int[2];
        // Center of the image/texture 
        imageCenter[0]= imageW/2;
        imageCenter[1]= imageH/2;
        
        // imageW/ image H contains the real width of the image we will use given the resolution. 
        //The resolution is generated once based on the maximum resolution.
        imageW = (int) (imageW*((max_res_factor/res_factor)));
        imageH = (int) (imageH*((max_res_factor/res_factor)));

        // sample on a plane through the origin of the volume data
        double max = volume.getMaximum();

        // Color that will be used as a result 
        TFColor pixelColor = new TFColor();
        // Auxiliar color
        TFColor colorAux;

        // Contains the voxel value of interest
        int val;
        
        // Iterate on every pixel.
        for (int j = imageCenter[1] - imageH/2; j < imageCenter[1] + imageH/2; j++) {
            for (int i =  imageCenter[0] - imageW/2; i <imageCenter[0] + imageW/2; i++) {
                
                // computes the pixelCoord which contains the 3D coordinates of the pixels (i,j)
                computePixelCoordinatesFloat(pixelCoord,volumeCenter,uVec,vVec,i,j);
                
                //we now have to get the value for the in the 3D volume for the pixel
                //we can use a nearest neighbor implementation like this:
//                val = (int) volume.getVoxelNN(pixelCoord);
                
                //you have also the function getVoxelLinearInterpolated in Volume.java          
//                val = (int) volume.getVoxelLinearInterpolate(pixelCoord);
                
                //you have to implement this function below to get the cubic interpolation
                val = (int) volume.getVoxelLinearInterpolate(pixelCoord);
                
                
                // Map the intensity to a grey value by linear scaling
//                 pixelColor.r = (val/max);
//                 pixelColor.g = pixelColor.r;
//                 pixelColor.b = pixelColor.r;

                // the following instruction makes intensity 0 completely transparent and the rest opaque
//                 pixelColor.a = val > 0 ? 1.0 : 0.0;   
                
                // Alternatively, apply the transfer function to obtain a color using the tFunc attribute
                 colorAux= tFunc.getColor(val);
                 pixelColor.r=colorAux.r;
                 pixelColor.g=colorAux.g;
                 pixelColor.b=colorAux.b;
                 pixelColor.a=colorAux.a; 
                // IMPORTANT: You can also simply use pixelColor = tFunc.getColor(val); However then you copy by reference and this means that if you change 
                // pixelColor you will be actually changing the transfer function So BE CAREFUL when you do this kind of assignments

                //BufferedImage/image/texture expects a pixel color packed as ARGB in an int
                //use the function computeImageColor to convert your double color in the range 0-1 to the format need by the image
                int pixelColor_i = computeImageColor(pixelColor.r,pixelColor.g,pixelColor.b,pixelColor.a);
                image.setRGB(i, j, pixelColor_i);
            }
        }
}
    

    
    //Do NOT modify this function
    //
    //Function that updates the "image" attribute using the MIP raycasting
    //It returns the color assigned to a ray/pixel given it's starting point (entryPoint) and the direction of the ray(rayVector).
    // exitPoint is the last point.
    //ray must be sampled with a distance defined by the sampleStep
   
    int traceRayMIP(double[] entryPoint, double[] exitPoint, double[] rayVector, double sampleStep) {
    	//compute the increment and the number of samples
        double[] increments = new double[3];
        VectorMath.setVector(increments, rayVector[0] * sampleStep, rayVector[1] * sampleStep, rayVector[2] * sampleStep);
        
        // Compute the number of times we need to sample
        double distance = VectorMath.distance(entryPoint, exitPoint);
        int nrSamples = 1 + (int) Math.floor(VectorMath.distance(entryPoint, exitPoint) / sampleStep);

        //the current position is initialized as the entry point
        double[] currentPos = new double[3];
        VectorMath.setVector(currentPos, entryPoint[0], entryPoint[1], entryPoint[2]);
       
        double maximum = 0;
        do {
            double value = volume.getVoxelLinearInterpolate(currentPos)/255.; 
            if (value > maximum) {
                maximum = value;
            }
            for (int i = 0; i < 3; i++) {
                currentPos[i] += increments[i];
            }
            nrSamples--;
        } while (nrSamples > 0);

        double alpha = maximum > 0 ? 1 : 0; // If the max voxel has a value make it fully opaque.
        double r, g, b;
        r = g = b = maximum;
        int color = computeImageColor(r,g,b,alpha);
        return color;
    }
    
          
    //////////////////////////////////////////////////////////////////////
    ///////////////// FUNCTION TO BE IMPLEMENTED /////////////////////////
    ////////////////////////////////////////////////////////////////////// 
    //Function that updates the "image" attribute using the Isosurface raycasting
    //It returns the color assigned to a ray/pixel given it's starting point (entryPoint) and the direction of the ray(rayVector).
    // exitPoint is the last point.
    //ray must be sampled with a distance defined by the sampleStep  
public int traceRayIso(double[] entryPoint, double[] exitPoint, double[] rayVector, double sampleStep) {
       
        //Initialization of the colors as floating point values
        double alpha = 0.0;
        double r = isoColor.r;
        double g = isoColor.g;
        double b = isoColor.b;
        
         // Compute the increments
        double[] increments = new double[3];
        VectorMath.setVector(increments, rayVector[0] * sampleStep, rayVector[1] * sampleStep, rayVector[2] * sampleStep);
        
        // Compute the number of times we need to sample.
        int nrSamples = 1 + (int) Math.floor(VectorMath.distance(entryPoint, exitPoint) / sampleStep);

        // The current position is initialized as the entry point.
        double[] currentPosition = new double[3];
        double[] nextPosition = new double[3];
        VectorMath.setVector(currentPosition, entryPoint[0], entryPoint[1], entryPoint[2]);
        VectorMath.setVector(nextPosition, entryPoint[0] + increments[0], entryPoint[1]+ increments[1], entryPoint[2] + increments[2]);
        
        do {
            float currentValue = volume.getVoxelLinearInterpolate(currentPosition);
            float nextValue = volume.getVoxelLinearInterpolate(nextPosition);
           
            // If the isovalue is between current value and next value, perform bisection to find exact position.
            if((Math.floor(currentValue) >= iso_value && Math.floor(nextValue) <  iso_value  )
            || (Math.floor(currentValue) <  iso_value && Math.floor(nextValue) >= iso_value )){
                
                // Set alpha to 1: this ray should map to a full color on the screen.
                alpha = 1;

                bisection_accuracy(currentPosition, increments, sampleStep, nextValue, currentValue, iso_value);

                if (shadingMode) {
                    TFColor color = new TFColor(r, g, b, alpha);
                    color = computePhongShading(color, gradients.getGradient(currentPosition), rayVector);
                    r = color.r;
                    g = color.g;
                    b = color.b;
                }
            } else {
                
                // Move once step further along the ray (with step size equal to increments).
                VectorMath.setVector(currentPosition, 
                        currentPosition[0] + increments[0],
                        currentPosition[1] + increments[1],
                        currentPosition[2] + increments[2]);
                VectorMath.setVector(nextPosition, 
                        nextPosition[0] + increments[0],
                        nextPosition[1] + increments[1],
                        nextPosition[2] + increments[2]);
                nrSamples--;

            }
        } while (nrSamples > 0 && alpha == 0);

        // Computes the color to display the current ray to.
        return computeImageColor(r, g, b, alpha);
    }
   
    //////////////////////////////////////////////////////////////////////
    ///////////////// FUNCTION TO BE IMPLEMENTED /////////////////////////
    //////////////////////////////////////////////////////////////////////
    // Given the current sample position, increment vector of the sample (vector from previous sample to current sample) and sample Step. 
    // Previous sample value and current sample value, isovalue value.
    // The function should search for a position where the iso_value passes that it is more precise.
    // TODO: Comment.
   public void  bisection_accuracy (double[] currentPosition, double[] increments, double sampleStep, float nextValue, float currentValue, 
           float iso_value) {
        
        // Calculate next position according to current position and increments.
        double[] nextPosition = new double[3];
        VectorMath.setVector(nextPosition, 
                currentPosition[0] + increments[0], 
                currentPosition[1] + increments[1], 
                currentPosition[2] + increments[2]);
       
        // Calculate the left and rightmost positions, where the left position corresponds to the one with lowest value.
        double[] leftPosition = (Math.floor(currentValue) < iso_value) ? currentPosition : nextPosition;   
        double[] rightPosition = (Math.floor(currentValue) < iso_value) ? nextPosition : currentPosition;
        
        // Perform binary search between left position and right position to find position where value equals isovalue.
        while(Math.floor(VectorMath.distance(leftPosition, rightPosition)) > 0) {
            
            // Take the middle of left and right position.
            double[] midPosition = new double[3];
            VectorMath.setVector(midPosition, 
                    (currentPosition[0] + nextPosition[0]) / 2,
                    (currentPosition[1] + nextPosition[1]) / 2,
                    (currentPosition[2] + nextPosition[2]) / 2);
            
            double midValue = volume.getVoxelLinearInterpolate(midPosition);
            
            // TODO: Do this with an epsilon? Or not necessary because we floor?
            if(Math.floor(midValue) == iso_value){
                 VectorMath.setVector(currentPosition, midPosition[0], midPosition[1], midPosition[2]);
                 return;
            }
            else {
                // The isovalue has not been located, if the current mid position is higher, go left otherwise go right.
                if(Math.floor(midValue) < iso_value) {
                    VectorMath.setVector(leftPosition, midPosition[0], midPosition[1], midPosition[2]);
                } else {
                    VectorMath.setVector(rightPosition, midPosition[0], midPosition[1], midPosition[2]);
                }
            }
        }
        
        // If the left and right vector are 'too close', we treat them as equal and set current position equal to one of them.
        VectorMath.setVector(currentPosition, 
            leftPosition[0],
            leftPosition[1],
            leftPosition[2]);
   }
    
    //////////////////////////////////////////////////////////////////////
    ///////////////// FUNCTION TO BE IMPLEMENTED /////////////////////////
    ////////////////////////////////////////////////////////////////////// 
    //Function that updates the "image" attribute using the compositing// accumulatted raycasting
    //It returns the color assigned to a ray/pixel given it's starting point (entryPoint) and the direction of the ray(rayVector).
    // exitPoint is the last point.
    //ray must be sampled with a distance defined by the sampleStep
   // TODO: Comment.
    TFColor computeCompositeColor1D(double[] currentPos, int nrSamples, double[] rayVector) {
        
        // Compute the color at this position on the ray.
        int value = (int) volume.getVoxelLinearInterpolate(currentPos);
        TFColor currentColor = tFunc.getColor(value);
        
        // Draw a new sample if we have insufficient samples and early ray termination criterion is not met.
        if(nrSamples >= 0 && currentColor.a < 0.999) {

            // Phong-shade the current color.
            currentColor = computePhongShading(currentColor, gradients.getGradient(currentPos), rayVector);
            
            // Reorientate the position to the next sample step along the ray.
            for (int i = 0; i < 3; i++) {
                currentPos[i] += rayVector[i];
            }
            nrSamples--;

            // Recursive call to the next color.
            TFColor nextColor = computeCompositeColor1D(currentPos, nrSamples, rayVector);
            
            // Compute the composite color of the current and the next color along the ray.
            return new TFColor(
                currentColor.a * currentColor.r + (1 - currentColor.a) * nextColor.r,
                currentColor.a * currentColor.g + (1 - currentColor.a) * nextColor.g,
                currentColor.a * currentColor.b + (1 - currentColor.a) * nextColor.b,
                currentColor.a + (1 - currentColor.a) * nextColor.a
            );

        // Otherwise, return the base case sample.
        } else {
            currentColor.r *=  currentColor.a;
            currentColor.g *=  currentColor.a;
            currentColor.b *=  currentColor.a;
            return currentColor;
        }
    }
    
    TFColor computeCompositeColor2D(double[] currentPos, int nrSamples, double[] rayVector) {
        
        // Compute the color at this position on the ray.
        int value = (int) volume.getVoxelLinearInterpolate(currentPos); // TODO: Should we use linear or tricube?
        TFColor currentColor = tFunc2D.color;
        double gradientMagnitude = this.gradients.getGradient(currentPos).mag;
        double currentOpacity = this.computeOpacity2DTF(tFunc2D.baseIntensity, tFunc2D.radius, value, gradientMagnitude) * currentColor.a;
        currentOpacity = silhouetteOpacity(currentOpacity, this.gradients.getGradient(currentPos));

        // Draw a new sample if we have insufficient samples and the opacity is not densed.
        if (nrSamples >= 0 && currentColor.a < 0.999) {
            
            // Phong-shade the current color.
            currentColor = computePhongShading(currentColor, gradients.getGradient(currentPos), rayVector);
                        
            // Reorientate the position to the next sample step along the ray.
            for (int i = 0; i < 3; i++) {
                currentPos[i] += rayVector[i];
            }
            nrSamples--;
            
            // Recursive call to the next color.
            TFColor nextColor = computeCompositeColor2D(currentPos, nrSamples, rayVector);   

            // Compute the composite color of the current and the next color along the ray.
            return new TFColor(
                currentOpacity * currentColor.r + (1 - currentOpacity) * nextColor.r,
                currentOpacity * currentColor.g + (1 - currentOpacity) * nextColor.g,
                currentOpacity * currentColor.b + (1 - currentOpacity) * nextColor.b,
                currentOpacity + (1 - currentOpacity) * nextColor.a
            );
        // Otherwise, return the current color: we either have enough samples or opacity is densed.
        } else {
            return new TFColor(0, 0, 0, 0);
        }
    }
    
    // Increases the opacity of volume samples where the gradient nears perpendicular to the view direction.
    // @param gradient      gradient object.
    // @param rayVector     the vector of light.
    // @return              a new opacity for the voxel.
    public double silhouetteOpacity(double currentOpacity, VoxelGradient gradient) {
        if(currentOpacity > 0 && silhouetteMode) {
            // Increase the opacity of volume samples where the gradient nears perpendicular to the view direction.
            double lightVectorDotNormGradientVector = gradient.x * normalLightVector[0] / gradient.mag + 
                    gradient.y * normalLightVector[0] / gradient.mag + gradient.z * normalLightVector[0] / gradient.mag;
            return Math.min(baseOpacity + Math.pow(1 - Math.abs(lightVectorDotNormGradientVector), sharpness), 0.9);
        }
        // If there is no opacity, then we do not want to 'add' opacity.
        return currentOpacity;
    }
    
    public int traceRayComposite(double[] entryPoint, double[] exitPoint, double[] rayVector, double sampleStep) {

        // Compute the nr of required samples.
        int nrOfSamples = 1 + (int) Math.floor(VectorMath.distance(entryPoint, exitPoint) / sampleStep);

        // The current position along the ray is the entry point.
        double[] currentPosition = new double[3];
        VectorMath.setVector(currentPosition, entryPoint[0], entryPoint[1], entryPoint[2]);
        
        TFColor voxel_color = new TFColor();
        
        // Depending on the mode, compute the color.
        if (compositingMode) {
            // 1D transfer function.
            voxel_color = computeCompositeColor1D(currentPosition, nrOfSamples, rayVector);
            
        } else if (tf2dMode) {
            // 2D transfer function.
            voxel_color = computeCompositeColor2D(currentPosition, nrOfSamples, rayVector);
        }
            
        // Computes the color
        return computeImageColor(voxel_color.r, voxel_color.g, voxel_color.b, voxel_color.a);
    }
    
    //////////////////////////////////////////////////////////////////////
    ///////////////// FUNCTION TO BE IMPLEMENTED /////////////////////////
    ////////////////////////////////////////////////////////////////////// 
    // Compute Phong Shading given the voxel color (material color), the gradient, the light vector and view vector 
    public TFColor computePhongShading(TFColor voxel_color, VoxelGradient gradient,
            double[] rayVector) {

        // Material property constants
        double ka = 0.1; // Ambient. 
        double kd = 0.7; // Diffuse.
        double ks = 0.2; // Specular.
        int alpha = 100; // Alpha.
        
        double lightLength = VectorMath.length(normalLightVector);
        double rayLength = VectorMath.length(rayVector);
        
        // No shading needed if mode is not on, if the color is black or white, or if the vectors have length 0.
        if(!shadingMode || (voxel_color.r == 0 && voxel_color.g == 0 && voxel_color.b == 0) 
                || lightLength == 0 || rayLength == 0 || gradient.mag == 0) {
            return voxel_color;
        }
        
        // Light color RGB (usually all 1.0,1.0,1.0, from slides).
        double[] La = {1, 1, 1};
        double[] Ld = {1, 1, 1};
        double[] Ls = {1, 1, 1};

        // Normalized vectors.
        double[] normGradientVector = {gradient.x / gradient.mag, gradient.y / gradient.mag, gradient.z / gradient.mag};
        double[] normRayVector = {rayVector[0] / rayLength, rayVector[1] / rayLength, rayVector[2] / rayLength};
        
        // cosTheta (gradient vector is our normal vector) 
        double cosTheta = VectorMath.dotproduct(normalLightVector, normGradientVector);

        // R
        double temp = VectorMath.dotproduct(normGradientVector, normalLightVector) * 2;
        double[] temptimesGradientVector = {normGradientVector[0] * temp, normGradientVector[1] * temp, normGradientVector[2] * temp};
        double[] R = {temptimesGradientVector[0] - normalLightVector[0], temptimesGradientVector[1] - normalLightVector[1], temptimesGradientVector[2] - normalLightVector[2]};
        double cosPhi = VectorMath.dotproduct(normRayVector, R);
        
        double r = La[0] * ka * voxel_color.r + Ld[0] * kd * voxel_color.r * cosTheta + Ls[0] * ks * voxel_color.r * Math.pow(cosPhi, alpha);
        double g = La[1] * ka * voxel_color.g + Ld[1] * kd * voxel_color.g * cosTheta + Ls[1] * ks * voxel_color.g * Math.pow(cosPhi, alpha);
        double b = La[2] * ka * voxel_color.b + Ld[2] * kd * voxel_color.b * cosTheta + Ls[2] * ks * voxel_color.b * Math.pow(cosPhi, alpha);
        double a = voxel_color.a;
      
        r = Math.max(0, Math.min(1, r));
        g = Math.max(0, Math.min(1, g));
        b = Math.max(0, Math.min(1, b));
       
        return new TFColor(r, g, b, a);
    }
    
    
    
    
    //////////////////////////////////////////////////////////////////////
    ///////////////// LIMITED MODIFICATION IS NEEDED /////////////////////
    ////////////////////////////////////////////////////////////////////// 
    // Implements the basic tracing of rays trough the image and given the
    // camera transformation
    // It calls the functions depending on the raycasting mode
  
    public void raycast(double[] viewMatrix) {

    	//data allocation
        double[] viewVec = new double[3];
        double[] uVec = new double[3];
        double[] vVec = new double[3];
        double[] pixelCoord = new double[3];
        double[] entryPoint = new double[3];
        double[] exitPoint = new double[3];
        
        // increment in the pixel domain in pixel units
        int increment = 1;
        // sample step in voxel units
        int sampleStep = 1;
        // reset the image to black
        resetImage();
        
        // vector uVec and vVec define the view plane, 
        // perpendicular to the view vector viewVec which is going from the view point towards the object
        // uVec contains the up vector of the camera in world coordinates (image vertical)
        // vVec contains the horizontal vector in world coordinates (image horizontal)
        getViewPlaneVectors(viewMatrix,viewVec,uVec,vVec);
        
        
        // The result of the visualization is saved in an image(texture)
        // we update the vector according to the resolution factor
        // If the resolution is 0.25 we will sample 4 times more points. 
        for(int k=0;k<3;k++) {
            uVec[k]=res_factor*uVec[k];
            vVec[k]=res_factor*vVec[k];
        }
        
       // We get the size of the image/texture we will be puting the result of the 
        // volume rendering operation.
        int imageW=image.getWidth();
        int imageH=image.getHeight();

        int[] imageCenter = new int[2];
        // Center of the image/texture 
        imageCenter[0]= imageW/2;
        imageCenter[1]= imageH/2;
        
        // imageW/ image H contains the real width of the image we will use given the resolution. 
        //The resolution is generated once based on the maximum resolution.
        imageW = (int) (imageW*((max_res_factor/res_factor)));
        imageH = (int) (imageH*((max_res_factor/res_factor)));
        
        //The rayVector is pointing towards the scene
        double[] rayVector = new double[3];
        rayVector[0]=-viewVec[0];rayVector[1]=-viewVec[1];rayVector[2]=-viewVec[2];
             
        // compute the volume center
        double[] volumeCenter = new double[3];
        computeVolumeCenter(volumeCenter);

        
        // ray computation for each pixel
        for (int j = imageCenter[1] - imageH/2; j < imageCenter[1] + imageH/2; j += increment) {
            for (int i =  imageCenter[0] - imageW/2; i <imageCenter[0] + imageW/2; i += increment) {
                // compute starting points of rays in a plane shifted backwards to a position behind the data set
            	computePixelCoordinatesBehindFloat(pixelCoord,viewVec,uVec,vVec,i,j);
            	// compute the entry and exit point of the ray
                computeEntryAndExit(pixelCoord, rayVector, entryPoint, exitPoint);
                if ((entryPoint[0] > -1.0) && (exitPoint[0] > -1.0)) {
                    int val = 0;
                    if (compositingMode || tf2dMode) {
                        val = traceRayComposite(entryPoint, exitPoint, rayVector, sampleStep);
                    } else if (mipMode) {
                        val = traceRayMIP(entryPoint, exitPoint, rayVector, sampleStep);
                    } else if (isoMode){
                        val= traceRayIso(entryPoint,exitPoint,rayVector, sampleStep);
                    }
                    for (int ii = i; ii < i + increment; ii++) {
                        for (int jj = j; jj < j + increment; jj++) {
                            image.setRGB(ii, jj, val);
                        }
                    }
                }

            }
        }
    }


//////////////////////////////////////////////////////////////////////
///////////////// FUNCTION TO BE IMPLEMENTED /////////////////////////
////////////////////////////////////////////////////////////////////// 
// Compute the opacity based on the value of the pixel and the values of the
// triangle widget tFunc2D contains the values of the baseintensity and radius
// tFunc2D.baseIntensity, tFunc2D.radius they are in image intensity units
// TODO: Elaborate, comment.
public double computeOpacity2DTF(double material_value, double material_r,
        double voxelValue, double gradMagnitude) {
    
    // Compute the angle of the voxel, determined by its gradient.
    double angleOfVoxel = Math.atan(Math.abs(voxelValue - material_value) / gradMagnitude);
    // Compute the angle set in the widget, relative to the highest gradient.
    double angleInWidget = Math.atan(material_r / gradients.getMaxGradientMagnitude());
    
    // If the angle set in the widget contains the angle of the current voxel, return full opacity.
    return angleOfVoxel < angleInWidget ? 1 : 0;
}  

  //////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////
  
    
    //Do NOT modify this function
    int computeImageColor(double r, double g, double b, double a){
		int c_alpha = 	a <= 1.0 ? (int) Math.floor(a * 255) : 255;
        int c_red = 	r <= 1.0 ? (int) Math.floor(r * 255) : 255;
        int c_green = 	g <= 1.0 ? (int) Math.floor(g * 255) : 255;
        int c_blue = 	b <= 1.0 ? (int) Math.floor(b * 255) : 255;
        int pixelColor = getColorInteger(c_red,c_green,c_blue,c_alpha);
        return pixelColor;
	}
    //Do NOT modify this function    
    public void resetImage(){
    	for (int j = 0; j < image.getHeight(); j++) {
	        for (int i = 0; i < image.getWidth(); i++) {
	            image.setRGB(i, j, 0);
	        }
	    }
    }
   
    //Do NOT modify this function
    void computeIncrementsB2F(double[] increments, double[] rayVector, double sampleStep) {
        // we compute a back to front compositing so we start increments in the oposite direction than the pixel ray
    	VectorMath.setVector(increments, -rayVector[0] * sampleStep, -rayVector[1] * sampleStep, -rayVector[2] * sampleStep);
    }
    
    //used by the slicer
    //Do NOT modify this function
    void getViewPlaneVectors(double[] viewMatrix, double viewVec[], double uVec[], double vVec[]) {
            VectorMath.setVector(viewVec, viewMatrix[2], viewMatrix[6], viewMatrix[10]);
	    VectorMath.setVector(uVec, viewMatrix[0], viewMatrix[4], viewMatrix[8]);
	    VectorMath.setVector(vVec, viewMatrix[1], viewMatrix[5], viewMatrix[9]);
	}
    
    //used by the slicer	
    //Do NOT modify this function
    void computeVolumeCenter(double volumeCenter[]) {
	VectorMath.setVector(volumeCenter, volume.getDimX() / 2, volume.getDimY() / 2, volume.getDimZ() / 2);
    }
	
    //used by the slicer
    //Do NOT modify this function
    void computePixelCoordinatesFloat(double pixelCoord[], double volumeCenter[], double uVec[], double vVec[], float i, float j) {
        // Coordinates of a plane centered at the center of the volume (volumeCenter and oriented according to the plane defined by uVec and vVec
            float imageCenter = image.getWidth()/2;
            pixelCoord[0] = uVec[0] * (i - imageCenter) + vVec[0] * (j - imageCenter) + volumeCenter[0];
            pixelCoord[1] = uVec[1] * (i - imageCenter) + vVec[1] * (j - imageCenter) + volumeCenter[1];
            pixelCoord[2] = uVec[2] * (i - imageCenter) + vVec[2] * (j - imageCenter) + volumeCenter[2];
    }
        
    //Do NOT modify this function
    void computePixelCoordinates(double pixelCoord[], double volumeCenter[], double uVec[], double vVec[], int i, int j) {
        // Coordinates of a plane centered at the center of the volume (volumeCenter and oriented according to the plane defined by uVec and vVec
            int imageCenter = image.getWidth()/2;
            pixelCoord[0] = uVec[0] * (i - imageCenter) + vVec[0] * (j - imageCenter) + volumeCenter[0];
            pixelCoord[1] = uVec[1] * (i - imageCenter) + vVec[1] * (j - imageCenter) + volumeCenter[1];
            pixelCoord[2] = uVec[2] * (i - imageCenter) + vVec[2] * (j - imageCenter) + volumeCenter[2];
    }
    //Do NOT modify this function
    void computePixelCoordinatesBehindFloat(double pixelCoord[], double viewVec[], double uVec[], double vVec[], float i, float j) {
            int imageCenter = image.getWidth()/2;
            // Pixel coordinate is calculate having the center (0,0) of the view plane aligned with the center of the volume and moved a distance equivalent
            // to the diaganal to make sure I am far away enough.

            double diagonal = Math.sqrt((volume.getDimX()*volume.getDimX())+(volume.getDimY()*volume.getDimY())+ (volume.getDimZ()*volume.getDimZ()))/2;               
            pixelCoord[0] = uVec[0] * (i - imageCenter) + vVec[0] * (j - imageCenter) + viewVec[0] * diagonal + volume.getDimX() / 2.0;
            pixelCoord[1] = uVec[1] * (i - imageCenter) + vVec[1] * (j - imageCenter) + viewVec[1] * diagonal + volume.getDimY() / 2.0;
            pixelCoord[2] = uVec[2] * (i - imageCenter) + vVec[2] * (j - imageCenter) + viewVec[2] * diagonal + volume.getDimZ() / 2.0;
    }
    //Do NOT modify this function
    void computePixelCoordinatesBehind(double pixelCoord[], double viewVec[], double uVec[], double vVec[], int i, int j) {
            int imageCenter = image.getWidth()/2;
            // Pixel coordinate is calculate having the center (0,0) of the view plane aligned with the center of the volume and moved a distance equivalent
            // to the diaganal to make sure I am far away enough.

            double diagonal = Math.sqrt((volume.getDimX()*volume.getDimX())+(volume.getDimY()*volume.getDimY())+ (volume.getDimZ()*volume.getDimZ()))/2;               
            pixelCoord[0] = uVec[0] * (i - imageCenter) + vVec[0] * (j - imageCenter) + viewVec[0] * diagonal + volume.getDimX() / 2.0;
            pixelCoord[1] = uVec[1] * (i - imageCenter) + vVec[1] * (j - imageCenter) + viewVec[1] * diagonal + volume.getDimY() / 2.0;
            pixelCoord[2] = uVec[2] * (i - imageCenter) + vVec[2] * (j - imageCenter) + viewVec[2] * diagonal + volume.getDimZ() / 2.0;
    }
	
    //Do NOT modify this function
    public int getColorInteger(int c_red, int c_green, int c_blue, int c_alpha) {
    	int pixelColor = (c_alpha << 24) | (c_red << 16) | (c_green << 8) | c_blue;
    	return pixelColor;
    } 
    //Do NOT modify this function
    public RaycastRenderer() {
        panel = new RaycastRendererPanel(this);
        panel.setSpeedLabel("0");
        isoColor = new TFColor();
        isoColor.r=1.0;isoColor.g=1.0;isoColor.b=0.0;isoColor.a=1.0;
    }
    
     
    //Do NOT modify this function
    public void setVolume(Volume vol) {
        System.out.println("Assigning volume");
        volume = vol;

        System.out.println("Computing gradients");
        gradients = new GradientVolume(vol);

        // set up image for storing the resulting rendering
        // the image width and height are equal to the length of the volume diagonal
        int imageSize = (int) Math.floor(Math.sqrt(vol.getDimX() * vol.getDimX() + vol.getDimY() * vol.getDimY()
                + vol.getDimZ() * vol.getDimZ())* (1/max_res_factor));
        if (imageSize % 2 != 0) {
            imageSize = imageSize + 1;
        }
        
        image = new BufferedImage(imageSize, imageSize, BufferedImage.TYPE_INT_ARGB);
      
                  
           
        // Initialize transferfunction 
        tFunc = new TransferFunction(volume.getMinimum(), volume.getMaximum());
        tFunc.setTestFunc();
        tFunc.addTFChangeListener(this);
        tfEditor = new TransferFunctionEditor(tFunc, volume.getHistogram());
        
        tFunc2D= new TransferFunction2D((short) (volume.getMaximum() / 2), 0.2*volume.getMaximum());
        tfEditor2D = new TransferFunction2DEditor(tFunc2D,volume, gradients);
        tfEditor2D.addTFChangeListener(this);

        System.out.println("Finished initialization of RaycastRenderer");
    }
    //Do NOT modify this function
    public RaycastRendererPanel getPanel() {
        return panel;
    }

    //Do NOT modify this function
    public TransferFunction2DEditor getTF2DPanel() {
        return tfEditor2D;
    }
    //Do NOT modify this function
    public TransferFunctionEditor getTFPanel() {
        return tfEditor;
    }
    //Do NOT modify this function
    public void setShadingMode(boolean mode) {
        shadingMode = mode;
        changed();
    }
    
    // Function setting silhouette mode to mode.
    public void setSilhouetteSettings(boolean mode, double b, double s) {
        silhouetteMode = mode;
        baseOpacity = b;
        sharpness = s;
        System.out.println("Set silhouette mode to " + mode + "," + b + ", " + s);
        changed();
    }
    
    // Function setting the normalized light vector.
    public void setNormalLightVector(double[] lightVector) {
        // If all values are zero, set default light vector.
        if(lightVector[0] == 0 && lightVector[1] == 0 && lightVector[2] == 0) {
            lightVector[1] = 1;
        }
        double norm = VectorMath.length(lightVector);
        VectorMath.setVector(normalLightVector, 
                lightVector[0] / norm, 
                lightVector[1] / norm, 
                lightVector[2] / norm);
        System.out.println(Arrays.toString(lightVector) + Arrays.toString(normalLightVector));
        changed();
    }
        
    //Do NOT modify this function
    public void setMIPMode() {
        setMode(false, true, false, false,false);
    }
    //Do NOT modify this function
    public void setSlicerMode() {
        setMode(true, false, false, false,false);
    }
    //Do NOT modify this function
    public void setCompositingMode() {
        setMode(false, false, true, false,false);
    }
    //Do NOT modify this function
    public void setTF2DMode() {
        setMode(false, false, false, true, false);
    }
    //Do NOT modify this function
    public void setIsoSurfaceMode(){
        setMode(false, false, false, false, true);
     }
    //Do NOT modify this function
    public void setIsoValue(float pIsoValue){
         iso_value = pIsoValue;
         if (isoMode){
             changed();
         }
             
     }
    //Do NOT modify this function
    public void setResFactor(int value) {
         float newRes= 1.0f/value;
         if (res_factor != newRes)
         {
             res_factor=newRes;
             if (volume != null) changed();
         }
     }
     
   //Do NOT modify this function
   public void setIsoColor(TFColor newColor)
     {
         this.isoColor.r=newColor.r;
         this.isoColor.g=newColor.g;
         this.isoColor.b=newColor.b;
         if ((volume!=null) && (this.isoMode)) changed();
     }
     
    //Do NOT modify this function
     public float getIsoValue(){
         return iso_value;
     }
    //Do NOT modify this function
    private void setMode(boolean slicer, boolean mip, boolean composite, boolean tf2d, boolean iso) {
        slicerMode = slicer;
        mipMode = mip;
        compositingMode = composite;
        tf2dMode = tf2d;        
        isoMode = iso;
        changed();
    }
    //Do NOT modify this function
    private boolean intersectLinePlane(double[] plane_pos, double[] plane_normal,
            double[] line_pos, double[] line_dir, double[] intersection) {

        double[] tmp = new double[3];

        for (int i = 0; i < 3; i++) {
            tmp[i] = plane_pos[i] - line_pos[i];
        }

        double denom = VectorMath.dotproduct(line_dir, plane_normal);
        if (Math.abs(denom) < 1.0e-8) {
            return false;
        }

        double t = VectorMath.dotproduct(tmp, plane_normal) / denom;

        for (int i = 0; i < 3; i++) {
            intersection[i] = line_pos[i] + t * line_dir[i];
        }

        return true;
    }
    //Do NOT modify this function
    private boolean validIntersection(double[] intersection, double xb, double xe, double yb,
            double ye, double zb, double ze) {

        return (((xb - 0.5) <= intersection[0]) && (intersection[0] <= (xe + 0.5))
                && ((yb - 0.5) <= intersection[1]) && (intersection[1] <= (ye + 0.5))
                && ((zb - 0.5) <= intersection[2]) && (intersection[2] <= (ze + 0.5)));

    }
    //Do NOT modify this function
    private void intersectFace(double[] plane_pos, double[] plane_normal,
            double[] line_pos, double[] line_dir, double[] intersection,
            double[] entryPoint, double[] exitPoint) {

        boolean intersect = intersectLinePlane(plane_pos, plane_normal, line_pos, line_dir,
                intersection);
        if (intersect) {

            double xpos0 = 0;
            double xpos1 = volume.getDimX();
            double ypos0 = 0;
            double ypos1 = volume.getDimY();
            double zpos0 = 0;
            double zpos1 = volume.getDimZ();

            if (validIntersection(intersection, xpos0, xpos1, ypos0, ypos1,
                    zpos0, zpos1)) {
                if (VectorMath.dotproduct(line_dir, plane_normal) < 0) {
                    entryPoint[0] = intersection[0];
                    entryPoint[1] = intersection[1];
                    entryPoint[2] = intersection[2];
                } else {
                    exitPoint[0] = intersection[0];
                    exitPoint[1] = intersection[1];
                    exitPoint[2] = intersection[2];
                }
            }
        }
    }
    
     
    
    //Do NOT modify this function
    void computeEntryAndExit(double[] p, double[] viewVec, double[] entryPoint, double[] exitPoint) {

        for (int i = 0; i < 3; i++) {
            entryPoint[i] = -1;
            exitPoint[i] = -1;
        }

        double[] plane_pos = new double[3];
        double[] plane_normal = new double[3];
        double[] intersection = new double[3];

        VectorMath.setVector(plane_pos, volume.getDimX(), 0, 0);
        VectorMath.setVector(plane_normal, 1, 0, 0);
        intersectFace(plane_pos, plane_normal, p, viewVec, intersection, entryPoint, exitPoint);

        VectorMath.setVector(plane_pos, 0, 0, 0);
        VectorMath.setVector(plane_normal, -1, 0, 0);
        intersectFace(plane_pos, plane_normal, p, viewVec, intersection, entryPoint, exitPoint);

        VectorMath.setVector(plane_pos, 0, volume.getDimY(), 0);
        VectorMath.setVector(plane_normal, 0, 1, 0);
        intersectFace(plane_pos, plane_normal, p, viewVec, intersection, entryPoint, exitPoint);

        VectorMath.setVector(plane_pos, 0, 0, 0);
        VectorMath.setVector(plane_normal, 0, -1, 0);
        intersectFace(plane_pos, plane_normal, p, viewVec, intersection, entryPoint, exitPoint);

        VectorMath.setVector(plane_pos, 0, 0, volume.getDimZ());
        VectorMath.setVector(plane_normal, 0, 0, 1);
        intersectFace(plane_pos, plane_normal, p, viewVec, intersection, entryPoint, exitPoint);

        VectorMath.setVector(plane_pos, 0, 0, 0);
        VectorMath.setVector(plane_normal, 0, 0, -1);
        intersectFace(plane_pos, plane_normal, p, viewVec, intersection, entryPoint, exitPoint);

    }

    //Do NOT modify this function
    private void drawBoundingBox(GL2 gl) {
        gl.glPushAttrib(GL2.GL_CURRENT_BIT);
        gl.glDisable(GL2.GL_LIGHTING);
        gl.glColor4d(1.0, 1.0, 1.0, 1.0);
        gl.glLineWidth(1.5f);
        gl.glEnable(GL.GL_LINE_SMOOTH);
        gl.glHint(GL.GL_LINE_SMOOTH_HINT, GL.GL_NICEST);
        gl.glEnable(GL.GL_BLEND);
        gl.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA);

        gl.glBegin(GL.GL_LINE_LOOP);
        gl.glVertex3d(-volume.getDimX() / 2.0, -volume.getDimY() / 2.0, volume.getDimZ() / 2.0);
        gl.glVertex3d(-volume.getDimX() / 2.0, volume.getDimY() / 2.0, volume.getDimZ() / 2.0);
        gl.glVertex3d(volume.getDimX() / 2.0, volume.getDimY() / 2.0, volume.getDimZ() / 2.0);
        gl.glVertex3d(volume.getDimX() / 2.0, -volume.getDimY() / 2.0, volume.getDimZ() / 2.0);
        gl.glEnd();

        gl.glBegin(GL.GL_LINE_LOOP);
        gl.glVertex3d(-volume.getDimX() / 2.0, -volume.getDimY() / 2.0, -volume.getDimZ() / 2.0);
        gl.glVertex3d(-volume.getDimX() / 2.0, volume.getDimY() / 2.0, -volume.getDimZ() / 2.0);
        gl.glVertex3d(volume.getDimX() / 2.0, volume.getDimY() / 2.0, -volume.getDimZ() / 2.0);
        gl.glVertex3d(volume.getDimX() / 2.0, -volume.getDimY() / 2.0, -volume.getDimZ() / 2.0);
        gl.glEnd();

        gl.glBegin(GL.GL_LINE_LOOP);
        gl.glVertex3d(volume.getDimX() / 2.0, -volume.getDimY() / 2.0, -volume.getDimZ() / 2.0);
        gl.glVertex3d(volume.getDimX() / 2.0, -volume.getDimY() / 2.0, volume.getDimZ() / 2.0);
        gl.glVertex3d(volume.getDimX() / 2.0, volume.getDimY() / 2.0, volume.getDimZ() / 2.0);
        gl.glVertex3d(volume.getDimX() / 2.0, volume.getDimY() / 2.0, -volume.getDimZ() / 2.0);
        gl.glEnd();

        gl.glBegin(GL.GL_LINE_LOOP);
        gl.glVertex3d(-volume.getDimX() / 2.0, -volume.getDimY() / 2.0, -volume.getDimZ() / 2.0);
        gl.glVertex3d(-volume.getDimX() / 2.0, -volume.getDimY() / 2.0, volume.getDimZ() / 2.0);
        gl.glVertex3d(-volume.getDimX() / 2.0, volume.getDimY() / 2.0, volume.getDimZ() / 2.0);
        gl.glVertex3d(-volume.getDimX() / 2.0, volume.getDimY() / 2.0, -volume.getDimZ() / 2.0);
        gl.glEnd();

        gl.glBegin(GL.GL_LINE_LOOP);
        gl.glVertex3d(-volume.getDimX() / 2.0, volume.getDimY() / 2.0, -volume.getDimZ() / 2.0);
        gl.glVertex3d(-volume.getDimX() / 2.0, volume.getDimY() / 2.0, volume.getDimZ() / 2.0);
        gl.glVertex3d(volume.getDimX() / 2.0, volume.getDimY() / 2.0, volume.getDimZ() / 2.0);
        gl.glVertex3d(volume.getDimX() / 2.0, volume.getDimY() / 2.0, -volume.getDimZ() / 2.0);
        gl.glEnd();

        gl.glBegin(GL.GL_LINE_LOOP);
        gl.glVertex3d(-volume.getDimX() / 2.0, -volume.getDimY() / 2.0, -volume.getDimZ() / 2.0);
        gl.glVertex3d(-volume.getDimX() / 2.0, -volume.getDimY() / 2.0, volume.getDimZ() / 2.0);
        gl.glVertex3d(volume.getDimX() / 2.0, -volume.getDimY() / 2.0, volume.getDimZ() / 2.0);
        gl.glVertex3d(volume.getDimX() / 2.0, -volume.getDimY() / 2.0, -volume.getDimZ() / 2.0);
        gl.glEnd();

        gl.glDisable(GL.GL_LINE_SMOOTH);
        gl.glDisable(GL.GL_BLEND);
        gl.glEnable(GL2.GL_LIGHTING);
        gl.glPopAttrib();


    }
    //Do NOT modify this function
    @Override
    public void visualize(GL2 gl) {

        double[] viewMatrix = new double[4 * 4];
        
        if (volume == null) {
            return;
        }
        	
         drawBoundingBox(gl);

        
     //    gl.glGetDoublev(GL2.GL_PROJECTION_MATRIX,viewMatrix,0);

         gl.glGetDoublev(GL2.GL_MODELVIEW_MATRIX, viewMatrix, 0);
                      
         
        long startTime = System.currentTimeMillis();
        if (slicerMode) {
            slicer(viewMatrix);    
        } else {
            raycast(viewMatrix);
        }
        
        long endTime = System.currentTimeMillis();
        double runningTime = (endTime - startTime);
        panel.setSpeedLabel(Double.toString(runningTime));

        Texture texture = AWTTextureIO.newTexture(gl.getGLProfile(), image, false);
        
        gl.glPushAttrib(GL2.GL_LIGHTING_BIT);
        gl.glDisable(GL2.GL_LIGHTING);
        //gl.glEnable(GL.GL_BLEND);
        //gl.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA);

        // draw rendered image as a billboard texture
        texture.enable(gl);
        texture.bind(gl);
        
        double halfWidth = res_factor*image.getWidth() / 2.0;
        gl.glPushMatrix();
        gl.glLoadIdentity();
        gl.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST);
        gl.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST);
        gl.glBegin(GL2.GL_QUADS);
        gl.glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
        gl.glTexCoord2d(texture.getImageTexCoords().left(), texture.getImageTexCoords().top());
        gl.glVertex3d(-halfWidth, -halfWidth, 0.0);
        gl.glTexCoord2d(texture.getImageTexCoords().left(), texture.getImageTexCoords().bottom());
        gl.glVertex3d(-halfWidth, halfWidth, 0.0);
        gl.glTexCoord2d(texture.getImageTexCoords().right(), texture.getImageTexCoords().bottom());
        gl.glVertex3d(halfWidth, halfWidth, 0.0);
        gl.glTexCoord2d(texture.getImageTexCoords().right(), texture.getImageTexCoords().top());
        gl.glVertex3d(halfWidth, -halfWidth, 0.0);
        gl.glEnd();
        texture.disable(gl);
        
        texture.destroy(gl);
        gl.glPopMatrix();

        gl.glPopAttrib();


        if (gl.glGetError() > 0) {
            System.out.println("some OpenGL error: " + gl.glGetError());
        }


    }
    
    //Do NOT modify this function
    public BufferedImage image;

    //Do NOT modify this function
    @Override
    public void changed() {
        for (int i=0; i < listeners.size(); i++) {
            listeners.get(i).changed();
        }
    }
    

}
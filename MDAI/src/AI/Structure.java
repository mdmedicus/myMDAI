package AI;

import java.util.ArrayList;

public class Structure {
	
	
	
	
	
	public ArrayList<float[]> layers;
	public ArrayList<float[]> weights;
	public ArrayList<Float> bias;
	public ArrayList<float[]> Vt;
	public ArrayList<float[]> St;
	public ArrayList<Float> bVt;
	public ArrayList<Float> bSt;
	public ArrayList<Integer> f_type;
	
	
	
	
	public float[] input;
	public int inputX, inputY, inputZ;
	
	
	public int filterNumber = 32;

	public int filterSize = 3;
	public int filterChannel = 1;
	public float[][] filters  = new float[filterNumber][filterSize*filterSize*filterChannel];
	public float[][] Vtf = new float[filterNumber][filterSize*filterSize*filterChannel];
	public float[][] Stf = new float[filterNumber][filterSize*filterSize*filterChannel];
	
	
	public int newSize;
	public float[][] pooling;
	
	
	public int[][] target;
	public float lr = 0.001f;
	public float b1 = 0.9f;
	public float b2 = 0.999f;
	public float ep = 0.00000001f;
	
	public Structure() {
		layers = new ArrayList<>();
		weights = new ArrayList<>();
		bias = new ArrayList<>();
		Vt = new ArrayList<>();
		St = new ArrayList<>();
		bVt = new ArrayList<>();
		bSt = new ArrayList<>();
		f_type = new ArrayList<>();
	}
}

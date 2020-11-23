package AI;

import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
import java.util.Arrays;
import java.util.Random;







public class Main {
	
	
	
////////////////////////////////////////////////mdmedicus///////////////////////////////////////////////////////
/////////////////////////////////////////gokhancetinkata95@gmail.com////////////////////////////////////////////
	
	
	Core core;
	static Structure str;
	

	
	boolean cnn = false;
	
	public Main(Structure str) {
		this.str = str;
	}
	
	public void ANN(int input) {
		str.layers.add(new float[input]);
	}	
	
	public void CNN() {
		str.newSize = str.inputX-str.filterSize+1;
		str.pooling = new float[str.filterNumber][str.newSize*str.newSize];
		str.layers.add(new float[str.newSize*str.newSize*str.filterNumber/4]);
		cnn = true;
	}
	
	public void addHidden(int size, int type) {
		str.layers.add(new float[size]);
		str.f_type.add(type);
	}
	
	
	public void create() {
				
		if(cnn) {
			for(int i = 0; i < str.filters.length; i++) {
				boolean isTrue = true;
				while(isTrue) {
					isTrue = false;
				float f;
				for(int a = 0; a < str.filterChannel; a++) {
				
				for(int m = 0; m < str.filterSize; m++) {
					for(int n = 0; n < str.filterSize; n++) {
						
						f =(new Random().nextInt(2));	
						if(f  == 0) {
							f=-1;
						}
						str.filters[i][m+n*str.filterSize+a*str.filterSize*str.filterSize] = f;
					}
				}
				}
			
				
				for(int z = 0; z < i; z++) {
					if(Arrays.equals(str.filters[i], str.filters[z])) {
						isTrue = true;
						break;
					}
				}
				}
			}
		}
		
		for(int i = 0; i < str.layers.size()-1; i++) {
			str.weights.add(new float[str.layers.get(i+1).length*str.layers.get(i).length]);
			str.bias.add(0f);
			str.St.add(new float[str.layers.get(i+1).length*str.layers.get(i).length]);
			str.Vt.add(new float[str.layers.get(i+1).length*str.layers.get(i).length]);

			str.bSt.add((float) .1);
			str.bVt.add((float) 0);
			
			//Xavier initialization
			float X_w =(float) ( Math.sqrt(6)/Math.sqrt(str.layers.get(i+1).length + str.layers.get(i).length));
			
			for(int m = 0; m < str.layers.get(i+1).length; m++) {
				int weightsNegative = 0;
				int weightsPositive = 0;
				for(int n = 0; n < str.layers.get(i).length; n++) {
					

					boolean again = true;
					float w = 0;
					while(again) {
						
					w =(new Random().nextInt(3)-1)*X_w;
					if(w != 0 && !((weightsNegative == str.layers.get(i).length/2) && (w < 0)) &&
							!((weightsPositive == str.layers.get(i).length/2 )&& (w > 0))) {
						again = false;
						if(w > 0) {
							weightsPositive++;
						} else {
							weightsNegative++;
						}
					} else if(w != 0 && (weightsNegative == str.layers.get(i).length/2) && (weightsPositive == str.layers.get(i).length/2 )) {
						again = false;
					}
					}
					str.weights.get(i)[m*str.layers.get(i).length+n] = w*100/(new Random().nextInt(100)+100);
					str.St.get(i)[m*str.layers.get(i).length+n] = 0;
					str.Vt.get(i)[m*str.layers.get(i).length+n] = 0;
				}
			}
			
			
		}
		
		core = new Core(str);
	}
	
	
	public int Training(int sayi) {
		
		
		if(cnn) {
			
			core.cnn();
			
		} else {
			for(int i = 0; i < str.layers.get(0).length; i++) {
				str.layers.get(0)[i] = str.input[i];
			}
		}
		
		
		
		for(int i = 0; i< str.layers.size()-1; i++) {
			if(str.f_type.get(i) == 0) {
				core.actN_relu(str.layers.get(i), str.layers.get(i+1), str.weights.get(i), str.bias.get(i));
			} else {
				core.actN_sigmoid(i+1);
			}

		}
		
		core.gradient(str.target[sayi]);
		core.op_adam();
//		core.op_rms(layers, weights, bias, Vt, bVt, target[sayi], 0.9, 0.00000001, 0.001,f_type);
		core.bpCNN();
		
		int bebe = 0;
		for(int i = 0; i < str.layers.get(str.layers.size()-1).length; i++) {
			if(i !=0) {
				if(str.layers.get(str.layers.size()-1)[i]>str.layers.get(str.layers.size()-1)[bebe]){
					bebe = i;
					
				}
			}
		}
		return bebe;
		
		
	}
	
	public float[] Test(int sayi) {
		
		
		if(cnn) {
			
			core.cnn();
			
		} else {
			for(int i = 0; i < str.layers.get(0).length; i++) {
				str.layers.get(0)[i] = str.input[i];
			}
		}
		
		
		float validationLoss = 0;
		
		
		for(int i = 0; i< str.layers.size()-1; i++) {
			if(str.f_type.get(i) == 0) {
				core.actN_relu(str.layers.get(i), str.layers.get(i+1), str.weights.get(i), str.bias.get(i));
			} else {
				core.actN_sigmoid(i+1);
			}

		}
		
		int bebe = 0;
		for(int i = 0; i < str.layers.get(str.layers.size()-1).length; i++) {
			if(i !=0) {
				if(str.layers.get(str.layers.size()-1)[i]>str.layers.get(str.layers.size()-1)[bebe]){
					bebe = i;
					
				}
			}
			validationLoss = validationLoss +
					(float)(str.target[sayi][i]-str.layers.get(str.layers.size()-1)[i])*(str.target[sayi][i]-str.layers.get(str.layers.size()-1)[i]);
		}
		
		return new float[] {bebe, validationLoss};
		
	}
	
	
	public void writeW() throws IOException {
		File w = new File("weights.txt");
		PrintStream ps = new PrintStream(w);
		for(int i = 0; i < str.layers.size()-1; i++) {
			for(int m = 0; m < str.layers.get(i+1).length; m++) {
				for(int n = 0; n < str.layers.get(i).length; n++) {
					ps.print(str.weights.get(i)[m*str.layers.get(i).length+n]+" ");
				}
			}
			ps.println("");
		}
	}
	
}

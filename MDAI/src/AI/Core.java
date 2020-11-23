package AI;

	import java.util.ArrayList;

	public class Core {
		
		float Cw = 0; //gradient of weights
		float Ca = 0; //gradient of activation

		ArrayList<float[]> Cz;
		Structure str;
		
		ArrayList<ArrayList<Integer>> bestOne;
		
		public Core(Structure str) {
			this.str = str;
			
			Cz = new ArrayList<>();
			bestOne = new ArrayList<>();
			for(int i = 1; i < str.layers.size(); i++) {
				Cz.add(new float[str.layers.get(i).length]);
			}
		
			
		}
		
		
		
		public void cnn() {
			bestOne = new ArrayList<>();
				for(int i= 0; i < str.filterNumber; i++) {
					
				for(int a = 0; a < str.filterChannel; a++) {
				for(int y = 0; y < str.newSize; y++) {
					for(int x = 0; x < str.newSize; x++) {
						float sonuc = 0f;
						for(int fy = 0; fy < str.filterSize; fy++) {
							for(int fx = 0; fx < str.filterSize; fx++) {
								sonuc +=
										str.input[x+fx+(y+fy)*str.inputX]*str.filters[i][a*str.filterSize*str.filterSize+fy*str.filterSize+fx];
							}
							
						}
						str.pooling[i][y*str.newSize + x] = sonuc+0.1f; //(str.filterSize*str.filterSize*str.filterChannel);
					}
				}
				
				}
				
				}
				
			
			for(int i = 0; i < str.filterNumber; i++) {
				ArrayList<Integer> eb = new ArrayList<>();
			for(int y = 0; y < str.newSize; y+=2) {
				for(int x = 0; x < str.newSize; x+=2 ) {
					int enBüyük = 0;
					str.layers.get(0)[i*(str.newSize/2)*(str.newSize/2) + y*(str.newSize/4) + x/2] =
							Math.max(Math.max(str.pooling[i][x+y*str.newSize], str.pooling[i][x+1+y*str.newSize]),
							Math.max(str.pooling[i][x+(y+1)*str.newSize], str.pooling[i][x+1+(y+1)*str.newSize]));
					
					if(str.layers.get(0)[i*(str.newSize/2)*(str.newSize/2) +y*(str.newSize/4) + x/2] == str.pooling[i][x+y*str.newSize]) {
						enBüyük = 0;
					} else if(str.layers.get(0)[i*(str.newSize/2)*(str.newSize/2) +y*(str.newSize/4) + x/2] == str.pooling[i][x+1+y*str.newSize]) {
						enBüyük = 1;
					} else if(str.layers.get(0)[i*(str.newSize/2)*(str.newSize/2) +y*(str.newSize/4) + x/2] == str.pooling[i][x+(y+1)*str.newSize]) {
						enBüyük = 2;
					} else if(str.layers.get(0)[i*(str.newSize/2)*(str.newSize/2) +y*(str.newSize/4) + x/2] == str.pooling[i][x+1+(y+1)*str.newSize]) {
						enBüyük = 3;
					}
					
					eb.add(enBüyük);
					
				}
			}
			
			bestOne.add(eb);
			}
			
		}
		
		
		//activation by sigmoid
		public void actN_sigmoid(int i) {
			
			for(int k = 0; k < str.layers.get(i).length; k++) {
				float z = 0;
				
				for(int j = 0; j < str.layers.get(i-1).length; j++) {
					z = z + str.layers.get(i-1)[j]*str.weights.get(i-1)[k*str.layers.get(i-1).length+j];
				}
				
				
				str.layers.get(i)[k] = (float)(1/(1+Math.exp(-z)));
				
				
			}
			
		}
		
		//activation by relu
		public void actN_relu(float[] inLayer, float[] outLayer, float[] weights, float bias) {
			
			for(int i = 0; i < outLayer.length; i++) {
				float z = bias;
				
				for(int j = 0; j < inLayer.length; j++) {
					z = z + inLayer[j]*weights[i*inLayer.length+j];
				}
				
				if(z > 0) {
					outLayer[i] = z;
				} else {
					outLayer[i] = 0;
				}
				
				
			}
			
		}
		
		
		
		//backpropagation by adam
		public void gradient(int[] target) {
			
			
			
//			long starttime00 = System.nanoTime();
			Cw = 0;
			Ca = 0;
			
//			System.out.println("0:"+(starttime00 - System.nanoTime()));
			
			
		//	long starttime0 = System.nanoTime();
			for(int m = 0; m < str.layers.get(str.layers.size()-1).length; m++) {
				
				if(str.f_type.get(str.f_type.size()-1) == 0) {
					
					if(str.layers.get(str.layers.size()-1)[m] > 0) {
						Cz.get(Cz.size()-1)[m] = (str.layers.get(str.layers.size()-1)[m]-target[m]);
					} else {
						Cz.get(Cz.size()-1)[m] = 0;
					}
				} else {

					Cz.get(Cz.size()-1)[m] = (str.layers.get(str.layers.size()-1)[m]-target[m])*
							str.layers.get(str.layers.size()-1)[m]*
							(1-str.layers.get(str.layers.size()-1)[m]);
				}
				
			
			}
//			System.out.println("1:"+(starttime0 - System.nanoTime()));
			
//			long startime = System.nanoTime();
			for(int i = 1; i < str.layers.size()-1; i++) {
				int layerNumber = str.layers.size()-1-i;
				
				for(int m = 0; m < str.layers.get(layerNumber).length; m++) {
					Ca = 0;
					
					if(str.f_type.get(layerNumber-1) == 0) {
						
						if(str.layers.get(layerNumber)[m] > 0) {
							for(int k = 0; k < str.layers.get(layerNumber+1).length; k ++) {
								Ca = Ca + (float)Cz.get(layerNumber)[k]*str.weights.get(layerNumber)[k*str.layers.get(layerNumber).length+m];
								
							}
							Cz.get(layerNumber)[m] = Ca;
							
						} else {
							Cz.get(layerNumber)[m] = 0;
						}
					} else {
						for(int k = 0; k < str.layers.get(layerNumber+1).length; k ++) {
							Ca = Ca + (float)Cz.get(layerNumber)[k]*str.weights.get(layerNumber)[k*str.layers.get(layerNumber).length+m];
							
						}
						Cz.get(layerNumber-1)[m] = Ca*(1-str.layers.get(layerNumber)[m])*str.layers.get(layerNumber)[m];
					}
						
					
				}
				
				
				
				
			}
			
			
			
			//Backpropagation of bias
	/*		for(int i = 1; i < layers.size(); i++) {
				int layerNumber = layers.size()-1-i;
				float Cb = 0;
				for(int a=0; a < Cz.get(i-1).length; a++) {
					Cb = Cb + Cz.get(i-1)[a];
				}
				
				bVt.set(layerNumber,b1*bVt.get(layerNumber) + (1-b1)*Cb);
				bSt.set(layerNumber, b2*bSt.get(layerNumber) + (1-b2)*Cb*Cb);
				
				float bmVt = bVt.get(layerNumber) /(1-Math.pow(b1, iteration));
				float bmSt = bSt.get(layerNumber) /(1-Math.pow(b2, iteration));
				
				bias.set(layerNumber, bias.get(layerNumber) - lr*bmVt/(Math.sqrt(bmSt)+ep));
				
				
			}
			*/

		}
		
		
		
		public void op_adam() {
			
				for(int i = 0; i < str.layers.size()-1; i++) {
				for(int m = 0; m < str.layers.get(i+1).length; m++) {
					for(int n = 0; n < str.layers.get(i).length; n++) {
						Cw = Cz.get(i)[m]*str.layers.get(i)[n];
						
						str.Vt.get(i)[m*str.layers.get(i).length+n] = str.b1*str.Vt.get(i)[m*str.layers.get(i).length+n] + (1-str.b1)*Cw;
						str.St.get(i)[m*str.layers.get(i).length+n] = str.b2*str.St.get(i)[m*str.layers.get(i).length+n] + (1-str.b2)*Cw*Cw;
						str.weights.get(i)[m*str.layers.get(i).length+n] = (float)(str.weights.get(i)[m*str.layers.get(i).length+n] - 
								str.lr*str.Vt.get(i)[m*str.layers.get(i).length+n]/(Math.sqrt(str.St.get(i)[m*str.layers.get(i).length+n])+str.ep));
						
						
						
					}
				}
			}
			
			
		}
		
		public void bpCNN() {
			float[] inputCz = new float[str.layers.get(0).length];
			float Ca = 0;
			for(int i = 0; i < str.layers.get(0).length; i++) {
				Ca = 0;
				for(int m = 0;  m < Cz.get(0).length; m++) {
					Ca += Cz.get(0)[m]*str.weights.get(0)[m*str.layers.get(0).length+i];
				}
				inputCz[i] = Ca;
			}
			
		float[][][][] Cf = new float[str.filterNumber][str.filterChannel][str.filterSize][str.filterSize];
		
		for(int i = 0; i < str.filterNumber; i++) {
			for(int a = 0; a < str.filterChannel; a++) {
			for(int yLayer = 0; yLayer < str.newSize/2; yLayer++) {
				for(int xLayer = 0; xLayer < str.newSize/2; xLayer++) {
					for(int yf = 0; yf < str.filterSize; yf++) {
						for(int xf = 0; xf < str.filterSize; xf++) {
							Cf[i][a][yf][xf] += inputCz[xLayer+yLayer*str.newSize/2]*str.input[(xLayer*2+yLayer*2*str.inputX)+
							              bestOne.get(i).get(xLayer+yLayer*str.newSize/2)%2+
							              str.inputX*((int)(bestOne.get(i).get(xLayer+yLayer*str.newSize/2)/2))+
							              xf+yf*str.inputX+a*str.filterSize*str.filterSize];
						}
					}
				}
			}
			}
		}	
			for(int i = 0; i < str.filterNumber; i++) {
				for(int a = 0; a < str.filterChannel; a++) {
			for(int y=0; y < str.filterSize; y++) {
				for(int x = 0 ; x < str.filterSize; x++) {
					
					str.Vtf[i][a*str.filterSize*str.filterSize+y*str.filterSize+x] = str.b1*str.Vtf[i][a*str.filterSize*str.filterSize+y*str.filterSize+x] + (1-str.b1)*Cf[i][a][y][x];
					str.Stf[i][a*str.filterSize*str.filterSize+y*str.filterSize+x] = str.b2*str.Stf[i][a*str.filterSize*str.filterSize+y*str.filterSize+x] + (1-str.b2)*Cf[i][a][y][x]*Cf[i][a][y][x];
					str.filters[i][a*str.filterSize*str.filterSize+y*str.filterSize+x] = str.filters[i][a*str.filterSize*str.filterSize+y*str.filterSize+x] - 
							(float)((str.lr/10)*str.Vtf[i][a*str.filterSize*str.filterSize+y*str.filterSize+x]/Math.sqrt(str.Stf[i][a*str.filterSize*str.filterSize+y*str.filterSize+x]+str.ep));
					
				}
			}
			}
		}
		}
		
		
		//backpropagation by RMSPROP
		public void op_rms(ArrayList<float[]> layers, ArrayList<float[][]> weights, ArrayList<Float> bias,
				ArrayList<float[][]> Vt, ArrayList<Float> bVt, int[] target, float b1, float ep, float lr,
				ArrayList<Integer> f_type) {
			
			
			
//			long starttime00 = System.nanoTime();
			Cz = new ArrayList<>();
			Cw = 0;
			Ca = 0;
			
			Cz.add(new float[layers.get(layers.size()-1).length]);
//			System.out.println("0:"+(starttime00 - System.nanoTime()));
			
			
		//	long starttime0 = System.nanoTime();
			for(int m = 0; m < layers.get(layers.size()-1).length; m++) {
				
				if(f_type.get(f_type.size()-1) == 0) {
					
					if(layers.get(layers.size()-1)[m] > 0) {
						Cz.get(0)[m] = (layers.get(layers.size()-1)[m]-target[m]);
					} else {
						Cz.get(0)[m] = 0;
					}
				} else {

					Cz.get(0)[m] = (layers.get(layers.size()-1)[m]-target[m])*layers.get(layers.size()-1)[m]*
							(1-layers.get(layers.size()-1)[m]);
				}
				
				for(int n = 0; n < layers.get(layers.size()-2).length; n++) {
					Cw = Cz.get(0)[m]*layers.get(layers.size()-2)[n];
					
					Vt.get(Vt.size()-1)[m][n] = (float)(0.9 * Vt.get(Vt.size()-1)[m][n] + 0.1*Cw*Cw);    
					
					
					weights.get(Vt.size()-1)[m][n] = (float)(weights.get(Vt.size()-1)[m][n]
							- ((lr*Cw)/(Math.sqrt(Vt.get(Vt.size()-1)[m][n])+0.00000001)));
					
					
				}
				
				
			
				
			}
//			System.out.println("1:"+(starttime0 - System.nanoTime()));
			
//			long startime = System.nanoTime();
			for(int i = 1; i < layers.size()-1; i++) {
				int layerNumber = layers.size()-1-i;
				Cz.add(new float[layers.get(layerNumber).length]);
				
				for(int m = 0; m < layers.get(layerNumber).length; m++) {
					Ca = 0;
					
					if(f_type.get(layerNumber-1) == 0) {
						
						if(layers.get(layerNumber)[m] > 0) {
							for(int k = 0; k < layers.get(layerNumber+1).length; k ++) {
								Ca = Ca + (float)Cz.get(i-1)[k]*weights.get(layerNumber)[k][m];
								
							}
							Cz.get(i)[m] = Ca;
							
						} else {
							Cz.get(i)[m] = 0;
						}
					} else {
						for(int k = 0; k < layers.get(layerNumber+1).length; k ++) {
							Ca = Ca + (float)Cz.get(i-1)[k]*weights.get(layerNumber)[k][m];
							
						}
						Cz.get(i)[m] = Ca*(1-layers.get(layerNumber)[m])*layers.get(layerNumber)[m];
					}
					
					
					
					
					
					for(int n = 0; n < layers.get(layerNumber-1).length; n++) {
						Cw = Cz.get(i)[m]*layers.get(layerNumber-1)[n];
						
						Vt.get(layerNumber-1)[m][n] = b1 * Vt.get(layerNumber-1)[m][n] + (1-b1)*Cw*Cw;
						
						weights.get(layerNumber-1)[m][n] = (float)(weights.get(layerNumber-1)[m][n]
								- ((lr*Cw)/(Math.sqrt(Vt.get(layerNumber-1)[m][n])+0.00000001)));
						
						
					}
					
					
					
					
				}
				
				
				
			}
			
			
			
			
		/*	//Backpropagation of bias
			for(int i = 1; i < layers.size(); i++) {
				int layerNumber = layers.size()-1-i;
				float Cb = 0;
				for(int a=0; a < Cz.get(i-1).length; a++) {
					Cb = Cb + Cz.get(i-1)[a];
				}
				
				bVt.set(layerNumber, b1*bVt.get(layerNumber) + (1-b1)*Cb*Cb);
				
				bias.set(layerNumber, (float)(bias.get(layerNumber) - (lr*Cb)/(Math.sqrt(bVt.get(layerNumber))+0.00000001)));
				
				
			}*/

		}
		//CNN
		
		
	}

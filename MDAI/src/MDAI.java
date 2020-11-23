import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;

import AI.Main;
import AI.Structure;

public class MDAI {
	
	
	
	

	///THIS PART IS FOR EXAMPLE ON MNIST

	static Main main;
	public static Structure str;

	static ArrayList<String> mnist_train = new ArrayList<>();

	static long startTime;

	long finishTime;
	
	static int validationDropOut = 0;
	static float validationOld = 100;
	
	static int trainingSize;
	static int yüzdeTraining;
	static int testSize;
	static int epochSize;
	
	static ArrayList<Integer> layerSizes;
	
	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		System.out.println("Hello");
		str = new Structure();
		main = new Main(str);
		layerSizes = new ArrayList<>();

		readConfig();
		
		main.ANN(str.inputX*str.inputY*str.inputZ);
	//	main.CNN();
		
		for(int i = 0; i < layerSizes.size(); i++) {
			main.addHidden(layerSizes.get(i), 1);
		}
		main.create();
		
		
		
		System.out.println("Deneme yapýlýyor...");
		//Deneme
		int kacinci = 0;
		

		BufferedReader br2;

		br2 = new BufferedReader(new FileReader("mnist_test.csv"));

		br2.readLine();

		String line = "";
		ArrayList<String> mnist_test = new ArrayList<>();
		
		while((line = br2.readLine()) != null) {
			mnist_test.add(line);
		}
		testSize = mnist_test.size();

		
		while(kacinci < epochSize) {
			
			Read();
			
			kacinci++;

			int doðruTraining = 0;
			int toplamTraining = 0;
			
			startTime = System.nanoTime();
			while( toplamTraining < trainingSize) {
				
				int whichNumber = new Random().nextInt(mnist_train.size());
				
				
				String[] allValues = mnist_train.get(whichNumber).split(",");
				mnist_train.remove(whichNumber);
				toplamTraining++;
				
				for(int i = 0; i < str.input.length; i++) {
					str.input[i] = (float)Integer.parseInt(allValues[i+1])/255;
				}
				
				
				
			/*	for(int i = 0; i < str.layers.get(0).length; i++) {
					str.layers.get(0)[i] = (float)Integer.parseInt(allValues[i+1])/255;
				}*/
				
		//		long time = System.nanoTime();
				int sonuc = main.Training(Integer.parseInt(allValues[0]));
		//		System.out.println("tra: "+ (time-System.nanoTime()));
				if(sonuc == Integer.parseInt(allValues[0])) {
					doðruTraining++;
				}
				
				if(toplamTraining%yüzdeTraining == 0) {

					System.out.println("%"+((int)toplamTraining*10/yüzdeTraining)+" tamamlandý."+(float)doðruTraining/toplamTraining);
				}
				
			}
			System.out.println(kacinci);
			System.out.println("Training Time : "+(startTime-System.nanoTime())/1000000000);
			System.out.println("Training Doðruluk Oraný: "+ (float)doðruTraining/toplamTraining);
		
		

			
			int doðru = 0;
			int toplam = 0;
			float validationLoss = 0;

			float[] sonuc;
			startTime = System.nanoTime();
			while(toplam < testSize) {
				
				String[] allValues = mnist_test.get(toplam).split(",");
				
				for(int i = 0; i < str.input.length; i++) {
					str.input[i] = (float)Integer.parseInt(allValues[i+1])/255;
				}
				
			/*	for(int a = 0; a < str.layers.get(0).length; a++) {
					str.layers.get(0)[a] = (float)Integer.parseInt(allValues[a+1])/255;
				}*/
				
				
				sonuc = main.Test(Integer.parseInt(allValues[0]));
				if(sonuc[0] == Integer.parseInt(allValues[0])) {
					doðru++;
				}
				toplam++;
				
				validationLoss = validationLoss + sonuc[1];
			}
			System.out.println("Test Time : "+(startTime-System.nanoTime())/1000000);

			validationLoss = validationLoss/toplam;
			
			System.out.println("Test doðruluk Oraný: "+(float)doðru/toplam);
			System.out.println("D/T: "+doðru+"/"+toplam);
			System.out.println("Validation Loss: "+ validationLoss);
			
		/*	if(validationLoss > validationOld) {
				validationDropOut++;
			//	main.lr = main.lr/2;
			} else {
				validationDropOut=0;
			}
			validationOld = validationLoss;
			if(validationDropOut == 2) {
			//	main.writeW();
			//	main.lr = main.lr*4;
				
			}*/
			
			
		}
		
		main.writeW();
		
		
	}
	
	public static void Read() throws IOException {
		BufferedReader br;
		String line = "";
		
		
		
		
		br = new BufferedReader(new FileReader("mnist_train.csv"));
		br.readLine();
		
		while((line = br.readLine()) != null) {
			mnist_train.add(line);
		}
		
		trainingSize = mnist_train.size();
		yüzdeTraining = trainingSize/10;
	}
	
	public static void readConfig() throws IOException {
		BufferedReader br_start = new BufferedReader(new FileReader("config.txt"));
		String configLine = "";
		String[] configString = new String[4];
		
		int z = 0;
		while((configLine = br_start.readLine()) != null) {
			configString[z] = configLine;
			z++;
		}
		
		
		str.inputX = Integer.parseInt(configString[0].split(" ")[0]);	
		str.inputY = Integer.parseInt(configString[0].split(" ")[1]);	
		str.inputZ = Integer.parseInt(configString[0].split(" ")[2]);	
		
		str.input = new float[str.inputX*str.inputY*str.inputZ];
		
		for(int i = 0; i < configString[1].split(" ").length; i++) {
			layerSizes.add(Integer.parseInt(configString[1].split(" ")[i]));
		}
		
		str.lr = Float.parseFloat(configString[2]);
		
		epochSize = Integer.parseInt(configString[3]);	
		
		 str.target = new int[layerSizes.get(layerSizes.size()-1)][layerSizes.get(layerSizes.size()-1)];
		 
		 for(int i = 0; i < str.target.length; i++) {
			 for(int m = 0; m < str.target.length; m++) {
				 if( i == m) {
					 str.target[i][m] = 1;
				 } else {
					 str.target[i][m] = 0;
				 }
			 }
		 }
		
		
	}
		
	public static void finish() {
		
	}
			
		
	}



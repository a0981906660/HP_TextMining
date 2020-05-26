import java.util.*;
import java.io.*;
import java.text.*;
import java.lang.*;

public class Penn2Ckip {

	public static void main(String[] args) {
		HashMap<String,String> map = loadMap("./map.txt");
		try{
			String oneLine = "";
			BufferedReader br = new BufferedReader(new FileReader(args[0]));
			while((oneLine = br.readLine()) != null)
			{
				String penn = oneLine.split(" ")[0];
				String ckip = oneLine.split(" ")[1];
				System.out.println("Process: " + penn + ", transformed and saved in " + ckip);
				BufferedReader brPenn = new BufferedReader(new FileReader(penn));
				FileWriter fwCkip= new FileWriter(ckip);
				
				String oneLine2 = "";
				while((oneLine2 = brPenn.readLine()) != null)
				{
					String[] tokens = oneLine2.split(" ");
					ArrayList<String> new_tokens = new ArrayList<String>();
					for(int i=0;i<tokens.length;i++)
					{
						String word = tokens[i].split("/")[0];
						String tag = tokens[i].split("/")[1];
						if(map.containsKey(tag))
							tag = map.get(tag);
						new_tokens.add(word + "(" + tag + ")");
					}
					if(tokens.length>0)
						fwCkip.write(new_tokens.get(0));
					if(tokens.length>1)
						for(int i=1;i<tokens.length;i++)
							fwCkip.write(" " + new_tokens.get(i));
					fwCkip.write("\n");
				}
				
				brPenn.close();
				fwCkip.close();
			}
			br.close();
			
		}catch(Exception e){System.out.println(e.toString());}		
	}

	private static HashMap<String, String> loadMap(String fn) {
		HashMap<String,String> map = new HashMap<String,String>();
		try{
			BufferedReader br = new BufferedReader(new FileReader(fn));
			String line = "";
			while((line = br.readLine()) != null)
			{
				String penn = line.split(",")[0];
				String ckip = line.split(",")[1];
				map.put(penn, ckip);
			}
			br.close();
		}catch(Exception e){System.out.println(e.toString());}	
		return map;
	}

}

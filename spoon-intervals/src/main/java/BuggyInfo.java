
public class BuggyInfo {
	public String targetFile;
	public int targetLine;
	public String desc;
	public BuggyInfo(String line){
		String[] res = line.split("#");
		this.targetLine = Integer.parseInt(res[1]);
		this.targetFile = res[0];
		if (res.length < 3) {
			this.desc = "not found";
		}
		else{
			this.desc = res[2];
		}
	}
}

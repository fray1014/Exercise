import org.junit.Test;

public class Xingye {
    public class Solution {
        /**
         * 将输入的十进制数字转换为对应的二进制字符串和十六进制字符串
         * @param number string字符串 十进制数字字符串
         * @return string字符串
         */
        public String changeFormatNumber (String number) {
            // write code here
//            int num = Integer.valueOf(number);
//            if(num>Math.pow(2,16)-1||num<-Math.pow(2,16)){
//                return "NODATA";
//            }
            try{
                int num = Integer.parseInt(number);
                if(num<-32768||num>32767)
                    return "NODATA";
                String binary = Integer.toBinaryString(num);
                StringBuffer binary2=new StringBuffer();
                if(binary.length()>16){
                    binary=binary.substring(16);
                }
                if(binary.length()<16){
                    for(int i=0;i<16-binary.length();i++){
                        binary2.append("0");
                    }
                }
                binary2.append(binary);
                String hex = Integer.toHexString(num).toUpperCase();
                StringBuffer hex2 = new StringBuffer();
                if(hex.length()>4){
                    hex=hex.substring(4);
                }
                if(hex.length()<4){
                    for(int i=0;i<4-hex.length();i++){
                        hex2.append("0");
                    }
                }
                hex2.append(hex);
                return binary2+","+hex2;
            }catch (NumberFormatException e){
                return "INPUTERROR";
            }

        }
    }
    @Test
    public void test(){
        Solution s = new Solution();
        System.out.println(s.changeFormatNumber("-32768"));
    }
}

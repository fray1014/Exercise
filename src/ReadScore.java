
import org.junit.Test;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.*;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.stream.Collectors;

public class ReadScore {
    @Test
    public void test(){
        try{
            String txtPath = "D:\\JAVA_Project\\exercise\\score.txt";
            Score s1 = readTxt2(txtPath);
//            txtPath = "D:\\JAVA_Project\\exercise\\T4_score.txt";
//            Score s2 = readTxt2(txtPath);
//            Score s3 = new Score();
//            s3.ld = Math.min(s1.ld,s2.ld);
//            s3.lu = Math.min(s1.lu,s2.lu);
//            s3.md = Math.min(s1.md,s2.md);
//            s3.mu = Math.min(s1.mu,s2.mu);
//            s3.rd = Math.min(s1.rd,s2.rd);
//            s3.ru = Math.min(s1.ru,s2.ru);
            System.out.println(s1);
        }catch (IOException e){
            e.printStackTrace();
        }
    }

    private static boolean sFlag = true;

    public static class Score{
        public double ld = 0;
        public double lu = 0;
        public double md = 0;
        public double mu = 0;
        public double rd = 0;
        public double ru = 0;
        public int cnt = 0;
    }
    private static Score sb1 = new Score();
    private static Score sb2 = new Score();
    public static Score[] readTxt(String txtPath) throws IOException {
        Path path = Paths.get(txtPath);
        List<String> lines = Files.readAllLines(path, StandardCharsets.UTF_8);
        lines.forEach(
                line -> {
                    if(line.isEmpty()){
                        sFlag = false;
                    }else{
                        List<Double> dlist = Arrays.stream(line.split(" ")).map(Double::valueOf).collect(Collectors.toList());
                        addScore(dlist,sFlag);
                    }
                }
        );
        Score[] res = new Score[2];
        getAvgScore();
        res[0] = sb1;
        res[1] = sb2;
        return res;
    }

    public static void addScore(List<Double> doubleList, boolean sFlag){
        double[] d = doubleList.stream().mapToDouble(Double::valueOf).toArray();
        if(sFlag){
            for(int i=0;i<d.length;i++){
                switch (i){
                    case 0:
                        sb1.ld+=d[i];
                        continue;
                    case 1:
                        sb1.lu+=d[i];
                        continue;
                    case 2:
                        sb1.md+=d[i];
                        continue;
                    case 3:
                        sb1.mu+=d[i];
                        continue;
                    case 4:
                        sb1.rd+=d[i];
                        continue;
                    case 5:
                        sb1.ru+=d[i];
                        continue;
                }
            }
            sb1.cnt++;
        }else{
            for(int i=0;i<d.length;i++){
                switch (i){
                    case 0:
                        sb2.ld+=d[i];
                        continue;
                    case 1:
                        sb2.lu+=d[i];
                        continue;
                    case 2:
                        sb2.md+=d[i];
                        continue;
                    case 3:
                        sb2.mu+=d[i];
                        continue;
                    case 4:
                        sb2.rd+=d[i];
                        continue;
                    case 5:
                        sb2.ru+=d[i];
                        continue;
                }
            }
            sb2.cnt++;
        }
    }

    public static void getAvgScore(){
        sb1.ld/=sb1.cnt;
        sb1.lu/=sb1.cnt;
        sb1.md/=sb1.cnt;
        sb1.mu/=sb1.cnt;
        sb1.rd/=sb1.cnt;
        sb1.ru/=sb1.cnt;
        sb2.ld/=sb2.cnt;
        sb2.lu/=sb2.cnt;
        sb2.md/=sb2.cnt;
        sb2.mu/=sb2.cnt;
        sb2.rd/=sb2.cnt;
        sb2.ru/=sb2.cnt;
    }

    public static Score readTxt2(String txtPath) throws IOException {
        Path path = Paths.get(txtPath);
        List<String> lines = Files.readAllLines(path, StandardCharsets.UTF_8);
        sb1.ld = 1;
        sb1.lu = 1;
        sb1.md = 1;
        sb1.mu = 1;
        sb1.rd = 1;
        sb1.ru = 1;
        lines.forEach(
                line -> {
                    double[] dlist = Arrays.stream(line.split(" ")).mapToDouble(Double::valueOf).toArray();
                    for(int i=0;i<dlist.length;i++){
                        switch (i){
                            case 0:
                                sb1.ld = Math.min(dlist[i],sb1.ld);
                                continue;
                            case 1:
                                sb1.lu = Math.min(dlist[i],sb1.lu);
                                continue;
                            case 2:
                                sb1.md = Math.min(dlist[i],sb1.md);
                                continue;
                            case 3:
                                sb1.mu = Math.min(dlist[i],sb1.mu);
                                continue;
                            case 4:
                                sb1.rd = Math.min(dlist[i],sb1.rd);
                                continue;
                            case 5:
                                sb1.ru = Math.min(dlist[i],sb1.ru);
                                continue;
                        }
                    }
                }
        );
        return sb1;
    }
}

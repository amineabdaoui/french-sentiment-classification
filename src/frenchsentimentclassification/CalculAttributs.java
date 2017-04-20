package frenchsentimentclassification;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Properties;
import java.util.StringTokenizer;
import static frenchsentimentclassification.ConstructionARFF.nbClassesFEEL;
import org.annolab.tt4j.TreeTaggerException;


/**
 *
 * @author amin.abdaoui
 */
public class CalculAttributs {
    
    ArrayList<String> alPosFEEL = new ArrayList<>();
    ArrayList<String> alNegFEEL = new ArrayList<>();
    ArrayList<String> alPosPolarimots = new ArrayList<>();
    ArrayList<String> alNegPolarimots = new ArrayList<>();
    ArrayList<String> alNeuPolarimots = new ArrayList<>();
    ArrayList<String> alPosAffects = new ArrayList<>();
    ArrayList<String> alNegAffects = new ArrayList<>();
    ArrayList<String> alNeuAffects = new ArrayList<>();
    ArrayList<String> alPosDiko = new ArrayList<>();
    ArrayList<String> alNegDiko = new ArrayList<>();
    ArrayList<String> alNeuDiko = new ArrayList<>();
    ArrayList<String> alZ = new ArrayList<>();
    ArrayList<ArrayList<String>> alZc = new ArrayList<>();
    ArrayList<ArrayList<String>> alEmoFEEL = new ArrayList<>();
    ArrayList<ArrayList<String>> alEmoAffects = new ArrayList<>();
    ArrayList<ArrayList<String>> alEmoDiko = new ArrayList<>();
    ArrayList<String> alWords = new ArrayList<>();
    private final ArrayList<String> Neg = new ArrayList<>();
    private LemmatiseurHandler lm;
    
    
    public CalculAttributs(Properties prop) throws FileNotFoundException, IOException{
        String line;
        BufferedReader r;
        // ClassLoader is used to load resources files from the jar
        ClassLoader classLoader = Thread.currentThread().getContextClassLoader();
        // FEEL
        if (prop.getProperty("Lexicons.feelPol").equalsIgnoreCase("yes") || prop.getProperty("Lexicons.feelEmo").equalsIgnoreCase("yes") || prop.getProperty("Lexicons.incongruity").equalsIgnoreCase("yes") || prop.getProperty("Lexicons.incongruityAll").equalsIgnoreCase("yes")){
            r = new BufferedReader(new InputStreamReader(new FileInputStream("ressources/FEEL.txt")));
            for (int i=1;i<=nbClassesFEEL;i++) alEmoFEEL.add(new ArrayList<String>());
            while ((line=r.readLine())!=null){
                // Polarité
                switch (line.split(";")[2]) {
                    case "positive":
                        alPosFEEL.add(line.split(";")[1].toLowerCase());
                        break;
                    case "negative":
                        alNegFEEL.add(line.split(";")[1].toLowerCase());
                        break;
                }
                // Emotion
                for (int i=0; i<nbClassesFEEL; i++) if (line.split(";")[i+3].equals("1")) alEmoFEEL.get(i).add(line.split(";")[1].toLowerCase());            
            }
            r.close();
        }
        // Polarimots
        if (prop.getProperty("Lexicons.polarimotsPol").equalsIgnoreCase("yes") || prop.getProperty("Lexicons.incongruity").equalsIgnoreCase("yes") || prop.getProperty("Lexicons.incongruityAll").equalsIgnoreCase("yes")){
            r = new BufferedReader(new InputStreamReader(new FileInputStream("ressources/Polarimots.txt")));
            while ((line=r.readLine())!=null){
                switch (line.split(";")[3]) {
                    case "positive":
                        alPosPolarimots.add(line.split(";")[1].toLowerCase());
                        break;
                    case "negative":
                        alNegPolarimots.add(line.split(";")[1].toLowerCase());
                        break;
                    case "neutre":
                        alNeuPolarimots.add(line.split(";")[1].toLowerCase());
                        break;
                }
            }
            r.close();
        }
        // Affects_Pol
        if (prop.getProperty("Lexicons.affectsPol").equalsIgnoreCase("yes") || prop.getProperty("Lexicons.incongruity").equalsIgnoreCase("yes") || prop.getProperty("Lexicons.incongruityAll").equalsIgnoreCase("yes")){
            r = new BufferedReader(new InputStreamReader(new FileInputStream("ressources/Augustin-pol.txt")));
            while ((line=r.readLine())!=null){
                switch (line.split(";")[1]) {
                    case "positive":
                        alPosAffects.add(line.split(";")[0].toLowerCase());
                        break;
                    case "negative":
                        alNegAffects.add(line.split(";")[0].toLowerCase());
                        break;
                    case "neutre":
                        alNeuAffects.add(line.split(";")[0].toLowerCase());
                        break;
                }
            }
            r.close();
        }
        // Affects_Emo
        if (prop.getProperty("Lexicons.affectsEmo").equalsIgnoreCase("yes")){
            r = new BufferedReader(new InputStreamReader(new FileInputStream("ressources/Augustin-emo.txt")));
            ArrayList<String> alClass = new ArrayList<>();
            while ((line=r.readLine())!=null){
                if (!alClass.contains(line.split(";")[1])){
                    alClass.add(line.split(";")[1]);
                    alEmoAffects.add(new ArrayList<String>());
                }
                alEmoAffects.get(alClass.indexOf(line.split(";")[1])).add(line.split(";")[0].toLowerCase());
            }
            r.close();
        }
        // Diko
        if (prop.getProperty("Lexicons.dikoPol").equalsIgnoreCase("yes") || prop.getProperty("Lexicons.incongruityAll").equalsIgnoreCase("yes")){
            r = new BufferedReader(new InputStreamReader(new FileInputStream("ressources/Diko.txt")));
            while ((line=r.readLine())!=null){
                switch (line.split(";")[2]) {
                    case "positive":
                        alPosDiko.add(line.split(";")[1].toLowerCase());
                        break;
                    case "negative":
                        alNegDiko.add(line.split(";")[1].toLowerCase());
                        break;
                    case "neutre":
                        alNeuDiko.add(line.split(";")[1].toLowerCase());
                        break;
                }
            }
            r.close();
        }
        // Diko_Emo
        if (prop.getProperty("Lexicons.dikoEmo").equalsIgnoreCase("yes")){
            r = new BufferedReader(new InputStreamReader(new FileInputStream("ressources/Diko-emo.txt")));
            ArrayList<String> alC = new ArrayList<>();
            while ((line=r.readLine())!=null){
                if (!alC.contains(line.split(";")[1])){
                    alC.add(line.split(";")[1]);
                    alEmoDiko.add(new ArrayList<String>());
                }
                alEmoDiko.get(alC.indexOf(line.split(";")[1])).add(line.split(";")[0].toLowerCase());
            }
            r.close();
        }
        // Negateurs
        r = new BufferedReader(new InputStreamReader(new FileInputStream("ressources/Negators.txt")));
        while ((line=r.readLine())!=null) Neg.add(line);
        r.close();
        // tt4j
        String ch="/data/TreeTagger";
        //if (prop.getProperty("TreeTagger.path")!=null) ch=prop.getProperty("TreeTagger.path");
        lm = new LemmatiseurHandler(ch);
    }

    public static int compte(String s, String chaine)
    {
        String str = new String(s);
        int cpt=0;
        while (str.contains(chaine)){
            str = str.substring(str.indexOf(chaine)+1);
            cpt++;
        }
        return cpt;
    }
    
    public int AllCaps(String tweet){
        int count=0;
        StringTokenizer st = new StringTokenizer(tweet, " 	.,;:'\"|()?!-_/<>‘’“”…«»•&#{[|`^@]}$*%1234567890", false);
        while (st.hasMoreElements()) if (isCap(st.nextToken())) count++;
        return count;
    }
    
    public boolean isCap(String word){
        boolean r = true;
        for (int i=0; i<word.length(); i++) if (Character.isUpperCase(word.charAt(i))==false) r=false;
        return r;
    }
    
    public int POS(String tweet, String pos) throws IOException, TreeTaggerException{
        int count=0;
        lm.clear();
        lm.setTermes(tweet);
        lm.process();
        for (String t:lm.getListPOS()){
            //t=t.split(":")[0];
            if (t.equals(pos)) count++;
        }
        return count;
    }
    
    public int ComputeIncongruity(String tweet){
        int count=0;
        String lemme, last="";
        StringTokenizer st = new StringTokenizer(tweet, " 	.,;:'\"|()?!-_/<>‘’“”…«»•&#{[|`^@]}$*%1234567890", false);
        while (st.hasMoreElements()){
            lemme=st.nextToken();
            if(lemme.contains("|")) lemme=lemme.split("|")[0];
            if (alPosFEEL.contains(lemme) || alPosAffects.contains(lemme) || alPosPolarimots.contains(lemme)){
                if (last.equals("neg")) count++;
                last="pos";
            }
            else if (alNegFEEL.contains(lemme) || alNegAffects.contains(lemme) || alNegPolarimots.contains(lemme)){
                if (last.equals("pos")) count++;
                last="neg";
            }
        }
        return count;
    }
    
    public int ComputeIncongruityAll(String tweet){
        int count=0;
        String lemme, last="";
        StringTokenizer st = new StringTokenizer(tweet, " 	.,;:'\"|()?!-_/<>‘’“”…«»•&#{[|`^@]}$*%1234567890", false);
        while (st.hasMoreElements()){
            lemme=st.nextToken();
            if(lemme.contains("|")) lemme=lemme.split("|")[0];
            if (alPosFEEL.contains(lemme) || alPosAffects.contains(lemme) || alPosPolarimots.contains(lemme) || alPosDiko.contains(lemme)){
                if (last.equals("neg")) count++;
                last="pos";
            }
            else if (alNegFEEL.contains(lemme) || alNegAffects.contains(lemme) || alNegPolarimots.contains(lemme) || alNegDiko.contains(lemme)){
                if (last.equals("pos")) count++;
                last="neg";
            }
        }
        return count;
    }
    
    
    public int ComputeEmotionFEEL(String tweet, int i) throws FileNotFoundException, IOException, TreeTaggerException{
        int count=0;
        String lemme;
        StringTokenizer st = new StringTokenizer(tweet, " 	.,;:'\"|()?!-_/<>‘’“”…«»•&#{[|`^@]}$*%1234567890", false);
        while (st.hasMoreElements()){
            lemme=st.nextToken();
            if(lemme.contains("|")) lemme=lemme.split("|")[0];
            if (alEmoFEEL.get(i).contains(lemme)) count++;
        }
        return count;
    }
    
    public int ComputeEmotionAffects(String tweet, int i) throws FileNotFoundException, IOException, TreeTaggerException{
        int count=0;
        String lemme;
        StringTokenizer st = new StringTokenizer(tweet, " 	.,;:'\"|()?!-_/<>‘’“”…«»•&#{[|`^@]}$*%1234567890", false);
        while (st.hasMoreElements()){
            lemme=st.nextToken();
            if(lemme.contains("|")) lemme=lemme.split("|")[0];
            if (alEmoAffects.get(i).contains(lemme)) count++;
        }
        return count;
    }
    
    public int ComputeEmotionDiko(String tweet, int i) throws FileNotFoundException, IOException, TreeTaggerException{
        int count=0;
        String lemme;
        StringTokenizer st = new StringTokenizer(tweet, " 	.,;:'\"|()?!-_/<>‘’“”…«»•&#{[|`^@]}$*%1234567890", false);
        while (st.hasMoreElements()){
            lemme=st.nextToken();
            if(lemme.contains("|")) lemme=lemme.split("|")[0];
            if (alEmoDiko.get(i).contains(lemme)) count++;
        }
        return count;
    }
    
    public int ComputePosFEEL(String tweet) throws FileNotFoundException, IOException, TreeTaggerException{
        int count=0;
        String lemme;
        StringTokenizer st = new StringTokenizer(tweet, " 	.,;:'\"|()?!-_/<>‘’“”…«»•&#{[|`^@]}$*%1234567890", false);
        while (st.hasMoreElements()){
            lemme=st.nextToken();
            if(lemme.contains("|")) lemme=lemme.split("|")[0];
            if (alPosFEEL.contains(lemme)) count++;
        }
        return count;
    }
    
    public int ComputeNegFEEL(String tweet) throws FileNotFoundException, IOException, TreeTaggerException{
        int count=0;
        String lemme;
        StringTokenizer st = new StringTokenizer(tweet, " 	.,;:'\"|()?!-_/<>‘’“”…«»•&#{[|`^@]}$*%1234567890", false);
        while (st.hasMoreElements()){
            lemme=st.nextToken();
            if(lemme.contains("|")) lemme=lemme.split("|")[0];
            if (alNegFEEL.contains(lemme)) count++;
        }
        return count;
    }
    
    public int ComputePosPolarimots(String tweet) throws FileNotFoundException, IOException, TreeTaggerException{
        int count=0;
        String lemme;
        StringTokenizer st = new StringTokenizer(tweet, " 	.,;:'\"|()?!-_/<>‘’“”…«»•&#{[|`^@]}$*%1234567890", false);
        while (st.hasMoreElements()){
            lemme=st.nextToken();
            if(lemme.contains("|")) lemme=lemme.split("|")[0];
            if (alPosPolarimots.contains(lemme)) count++;
        }
        return count;
    }
    
    public int ComputeNegPolarimots(String tweet) throws FileNotFoundException, IOException, TreeTaggerException{
        int count=0;
        String lemme;
        StringTokenizer st = new StringTokenizer(tweet, " 	.,;:'\"|()?!-_/<>‘’“”…«»•&#{[|`^@]}$*%1234567890", false);
        while (st.hasMoreElements()){
            lemme=st.nextToken();
            if(lemme.contains("|")) lemme=lemme.split("|")[0];
            if (alNegPolarimots.contains(lemme)) count++;
        }
        return count;
    } 
    
    public int ComputeNeuPolarimots(String tweet) throws FileNotFoundException, IOException, TreeTaggerException{
        int count=0;
        String lemme;
        StringTokenizer st = new StringTokenizer(tweet, " 	.,;:'\"|()?!-_/<>‘’“”…«»•&#{[|`^@]}$*%1234567890", false);
        while (st.hasMoreElements()){
            lemme=st.nextToken();
            if(lemme.contains("|")) lemme=lemme.split("|")[0];
            if (alNeuPolarimots.contains(lemme)) count++;
        }
        return count;
    }
    
    public int ComputePosAffects(String tweet) throws FileNotFoundException, IOException, TreeTaggerException{
        int count=0;
        String lemme;
        StringTokenizer st = new StringTokenizer(tweet, " 	.,;:'\"|()?!-_/<>‘’“”…«»•&#{[|`^@]}$*%1234567890", false);
        while (st.hasMoreElements()){
            lemme=st.nextToken();
            if(lemme.contains("|")) lemme=lemme.split("|")[0];
            if (alPosAffects.contains(lemme)) count++;
        }
        return count;
    }
    
    public int ComputeNegAffects(String tweet) throws FileNotFoundException, IOException, TreeTaggerException{
        int count=0;
        String lemme;
        StringTokenizer st = new StringTokenizer(tweet, " 	.,;:'\"|()?!-_/<>‘’“”…«»•&#{[|`^@]}$*%1234567890", false);
        while (st.hasMoreElements()){
            lemme=st.nextToken();
            if(lemme.contains("|")) lemme=lemme.split("|")[0];
            if (alNegAffects.contains(lemme)) count++;
        }
        return count;
    } 
    
    public int ComputeNeuAffects(String tweet) throws FileNotFoundException, IOException, TreeTaggerException{
        int count=0;
        String lemme;
        StringTokenizer st = new StringTokenizer(tweet, " 	.,;:'\"|()?!-_/<>‘’“”…«»•&#{[|`^@]}$*%1234567890", false);
        while (st.hasMoreElements()){
            lemme=st.nextToken();
            if(lemme.contains("|")) lemme=lemme.split("|")[0];
            if (alNeuAffects.contains(lemme)) count++;
        }
        return count;
    }
    
    public int ComputePosDiko(String tweet) throws FileNotFoundException, IOException, TreeTaggerException{
        int count=0;
        String lemme;
        StringTokenizer st = new StringTokenizer(tweet, " 	.,;:'\"|()?!-_/<>‘’“”…«»•&#{[|`^@]}$*%1234567890", false);
        while (st.hasMoreElements()){
            lemme=st.nextToken();
            if(lemme.contains("|")) lemme=lemme.split("|")[0];
            if (alPosDiko.contains(lemme)) count++;
        }
        return count;
    }
    
    public int ComputeNegDiko(String tweet) throws FileNotFoundException, IOException, TreeTaggerException{
        int count=0;
        String lemme;
        StringTokenizer st = new StringTokenizer(tweet, " 	.,;:'\"|()?!-_/<>‘’“”…«»•&#{[|`^@]}$*%1234567890", false);
        while (st.hasMoreElements()){
            lemme=st.nextToken();
            if(lemme.contains("|")) lemme=lemme.split("|")[0];
            if (alNegDiko.contains(lemme)) count++;
        }
        return count;
    } 
    
    public int ComputeNeuDiko(String tweet) throws FileNotFoundException, IOException, TreeTaggerException{
        int count=0;
        String lemme;
        StringTokenizer st = new StringTokenizer(tweet, " 	.,;:'\"|()?!-_/<>‘’“”…«»•&#{[|`^@]}$*%1234567890", false);
        while (st.hasMoreElements()){
            lemme=st.nextToken();
            if(lemme.contains("|")) lemme=lemme.split("|")[0];
            if (alNeuDiko.contains(lemme)) count++;
        }
        return count;
    }
    
    public boolean LastTokenEmoticone(String tweet) throws IOException{
        Charset charset = Charset.forName("Windows-1252");
        String tok [] = tweet.split("\t ");
        String lasttoken = tok[tok.length-1] ;
        if (lasttoken.equals("lienHTTP")) lasttoken = tok[tok.length-2];
        ClassLoader classLoader = Thread.currentThread().getContextClassLoader();
        BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream("ressources/emoticone.txt")));
        String ligne;
        while ((ligne=br.readLine())!=null){
                String [] tmp = ligne.split("\t");
                if(lasttoken.toUpperCase().indexOf(tmp[0].toUpperCase()) != -1){
                        br.close();
                        return true ;
                }
        }
        br.close();

        return false;
    }
    
    /**
     * TODO: est-ce que c'est bon d'ouvrir le fichier à chaque itération de while ?
     * 
     * @param tweet
     * @return
     * @throws IOException 
     */
    public boolean EmoticonesPos(String tweet) throws IOException{
        Charset charset = Charset.forName("Windows-1252");
        StringTokenizer st = new StringTokenizer(tweet, "\t ");
        ClassLoader classLoader = Thread.currentThread().getContextClassLoader();
        while (st.hasMoreElements()) {
                String token = (String) st.nextElement() ;
                BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream("ressources/emoticone.txt")));
                String ligne;	
                while ((ligne=br.readLine())!=null){
                        String [] tmp = ligne.split("\t");
                        if(token.indexOf(tmp[0]) != -1 && Double.parseDouble(tmp[1])>0){
                                br.close();
                                return true ;
                        }
                }
                br.close();
        }
        return false;
    }
    
    public boolean EmoticonesNeg(String tweet) throws IOException{
        Charset charset = Charset.forName("Windows-1252");
        StringTokenizer st = new StringTokenizer(tweet, "\t ");
        ClassLoader classLoader = Thread.currentThread().getContextClassLoader();
        while (st.hasMoreElements()) {
                String token = (String) st.nextElement() ;
                BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream("ressources/emoticone.txt")));
                String ligne;	
                while ((ligne=br.readLine())!=null){
                        String [] tmp = ligne.split("\t");
                        if(token.indexOf(tmp[0]) != -1 && Double.parseDouble(tmp[1])<0){
                                br.close();
                                return true ;
                        }
                }
                br.close();
        }
        return false;
    }
    
    public int ContiguousSequences(String tweet){
        int cpt = 0 ;
        int sqc = 0 ;
        int ltw = tweet.length() ;
        char tmp =' ';
        for(int i=0; i<ltw; i++){
                if( ((tweet.charAt(i) == '?') || (tweet.charAt(i) == '!')) && (cpt == 1) ){
                        sqc++ ;
                        cpt++ ;
                }else if( ((tweet.charAt(i) == '?') || (tweet.charAt(i) == '!')) && ((tmp!='?') && (tmp!='!')) ){
                        cpt = 1 ;
                }else{
                        cpt++;
                }
                tmp = tweet.charAt(i);
        }

        return sqc ;
    }
    
    public boolean Punctuation(String tweet){
        return tweet.contains("?") || tweet.contains("!");
    }
    
    public boolean LastTokenPonctuation(String tweet){
        StringTokenizer st = new StringTokenizer(tweet, ".,|;:'\"()-\t ");
        String lastToken = "", sauv="";
        while (st.hasMoreElements()) {
                sauv=lastToken;
                lastToken = (String) st.nextElement();
        }
        if (lastToken.equals("lienHTTP")) lastToken=sauv;
        for(int i=0 ; i<lastToken.length() ; i++){
                if((lastToken.charAt(i) == '?') || (lastToken.charAt(i) == '!')){
                        return true ;
                }
        }
        return false; 
    }
    
    public int CountHashtag (String tweet){
        int nbh = 0 ;
        for(int i=0 ; i<tweet.length(); i++){
                if(tweet.charAt(i) == '#')
                        nbh++ ;
        }
        return nbh;
    }

    public int ElongatedWords(String tweet){
        StringTokenizer st = new StringTokenizer(tweet, " 	.,;:'\"|()?!-1234567890");
        String lastToken = null ;
        int cpt = 0, elw = 0 ; 
        char tmp = ' ';
        while (st.hasMoreElements()) {
            lastToken = (String) st.nextElement() ;
            int ltw = lastToken.length();
            tmp = ' ';
            cpt = 0 ;
            for(int i=0; i<ltw; i++){
                if( lastToken.charAt(i) == tmp ){
                        cpt++ ;
                }else 
                        cpt = 0 ;

                if( cpt >= 2 ){
                        elw++ ;
                        break ;
                }
                tmp = lastToken.charAt(i);
            }
        }
        return elw ;
    }
        
    
    public boolean Negation(String tweet){
        boolean res=false;
        StringTokenizer st = new StringTokenizer(tweet, " 	.,;:\'\"|()?!-1234567890", false);
        while (st.hasMoreElements()) if (Neg.contains(st.nextElement())) res=true;
        return res;
    }
    
    public int CountNegation(String tweet){
        int cpt=0;
        StringTokenizer st = new StringTokenizer(tweet, " 	.,;:\'\"|()?!-1234567890", false);
        while (st.hasMoreElements()) if (Neg.contains(st.nextElement())) cpt++;
        return cpt;
    }
    
    public int nbApp(String line, char c){
        int count=0;
        for (int i=0; i<line.length(); i++) if (line.charAt(i)==c) count++;
        return count;
    }
    
    public String Lemmatiser(String tweet) throws Exception {
        String tweet_lem="";
        lm.clear();
        lm.setTermes(tweet);
        lm.process();
        for (String t:lm.getListTermeLem()){
            if (tweet_lem.length()>1) tweet_lem+=" ";
            if (t.contains("|") && t.length()>1){
                try{
                    t=t.split("\\|")[0];
                } catch(Exception e){
                }
            }
            tweet_lem+=t;
        }
        return tweet_lem;
    }
        
}
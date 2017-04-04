package frenchsentimentclassification;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Properties;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.unsupervised.attribute.StringToWordVector;

/**
 *
 * @author Amine
 */
public class SearchBestConfigurations {
    
    private ArrayList<Instances> trains;
    private ArrayList<Instances> tests;
    private ArrayList<Instances> trainsNgrams;
    private ArrayList<Instances> testsNgrams;
    
    public SearchBestConfigurations(ArrayList<Instances> trs, ArrayList<Instances> tss){
        trains = new ArrayList<Instances>(trs);
        tests = new ArrayList<Instances>(tss);
    }
    
    public PropWithMeasure bestNgrams(Properties prop, PrintWriter Out) throws IOException, Exception{
        ArrayList<Properties> alProp = new ArrayList();
        ArrayList<Double> alMiF = new ArrayList();
        int min=Integer.parseInt(prop.getProperty("Ngrams.min"));
        int max=Integer.parseInt(prop.getProperty("Ngrams.max"));
       
        System.out.println(min+" "+max);
        
        double res;
        for (int i=min; i<=max; i++){
            for (int j=i; j<=max; j++){
                System.out.println(i+" "+j);
                FileOutputStream out = new FileOutputStream("test/config"+i+j+".properties");
                Properties propTampon = new Properties(prop);
                propTampon.setProperty("Ngrams.min", String.valueOf(i));
                propTampon.setProperty("Ngrams.max", String.valueOf(j));
                propTampon.store(out, null);
                out.close();
                res = runNgrams(propTampon);
                alProp.add(new Properties(propTampon));
                alMiF.add(res);
                System.out.println(i+", "+j+" = "+res);
                Out.println(i+", "+j+" = "+res);
                Out.flush();
            }
        }
        // Find the best configuration
        double bestRes = Collections.max(alMiF);
        int bestIndex = alMiF.indexOf(bestRes);
        Out.println("Best config Ngram: "+alProp.get(bestIndex).getProperty("Ngrams.min")+" "+alProp.get(bestIndex).getProperty("Ngrams.max"));
        Out.flush();
        
        try (FileOutputStream out = new FileOutputStream("test/config.properties")) {
            prop.setProperty("Ngrams.min", alProp.get(bestIndex).getProperty("Ngrams.min"));
            prop.setProperty("Ngrams.max", alProp.get(bestIndex).getProperty("Ngrams.max"));
            prop.store(out, null);
        }
        System.out.println(" => Best: "+alProp.get(bestIndex).getProperty("Ngrams.min")+" "+alProp.get(bestIndex).getProperty("Ngrams.max"));
        setInstancesNgrams(prop);
        PropWithMeasure PM = new PropWithMeasure(prop, alMiF.get(bestIndex));
        
        return PM;
    }
    
    private void setInstancesNgrams(Properties props) throws Exception{
        Instances train, test;
        trainsNgrams = new ArrayList<Instances>();
        testsNgrams = new ArrayList<Instances>();
        for (int i=0; i<trains.size(); i++){
            train = new Instances(trains.get(i));
            test = new Instances(tests.get(i));
            StringToWordVector filter = Tokenisation.WordNgrams(props);
            filter.setInputFormat(train);
            train = Filter.useFilter(train, filter);
            test = Filter.useFilter(test, filter);
            train.setClass(train.attribute("_class"));
            test.setClass(train.attribute("_class"));
            trainsNgrams.add(train);
            testsNgrams.add(test);
        }
    }
    
    public double runNgrams(Properties props) throws FileNotFoundException, IOException, Exception{
        Instances train, test;
        double miF=0;
        for (int i=0; i<trains.size(); i++){
            train = new Instances(trains.get(i));
            test = new Instances(tests.get(i));
            StringToWordVector filter = Tokenisation.WordNgrams(props);
            filter.setInputFormat(train);
            train = Filter.useFilter(train, filter);
            test = Filter.useFilter(test, filter);
            train.setClass(train.attribute("_class"));
            test.setClass(train.attribute("_class"));
            SMO classifier = new SMO();
            classifier.buildClassifier(train);
            Evaluation eTest = new Evaluation(train);
            eTest.evaluateModel(classifier, test);
            miF += eTest.unweightedMicroFmeasure();
        }
        return miF/trains.size();
    }
    
    public double AttributeSelection(Properties props, double measure, PrintWriter Out) throws FileNotFoundException, IOException, Exception{
        Instances train, test;
        System.out.println("Without attribute selection = "+measure);
        double miF=0;
        ArrayList<Instances> NewTrains = new ArrayList<Instances>();
        ArrayList<Instances> NewTests = new ArrayList<Instances>();
        for (int i=0; i<trains.size(); i++){
            train = new Instances(trainsNgrams.get(i));
            test = new Instances(testsNgrams.get(i));
            AttributeSelection filter = SelectionAttributs.InfoGainAttributeEval(train);
            train = Filter.useFilter(train, filter);
            test = Filter.useFilter(test, filter);
            train.setClass(train.attribute("_class"));
            test.setClass(train.attribute("_class"));
            NewTrains.add(train);
            NewTests.add(test);
            SMO classifier = new SMO();
            classifier.buildClassifier(train);
            Evaluation eTest = new Evaluation(train);
            eTest.evaluateModel(classifier, test);
            miF += eTest.unweightedMicroFmeasure();
        }
        miF = miF/trains.size();
        System.out.println("With attribute selection = "+miF);
        Out.println("With attribute selection = "+miF);
        Out.flush();
        if (miF>measure){
            System.out.println(" => Selected");
            Out.println(" => Selected");
            trainsNgrams = NewTrains;
            testsNgrams = NewTests;
            try (FileOutputStream out = new FileOutputStream("test/config.properties")) {
            props.setProperty("FeatureSelection.InfoGain", "yes");
            props.store(out, null);
        }
        } else {
            System.out.println(" => Not selected");
            Out.println(" => Not selected");
        }
        Out.flush();
        return miF;
    }
    
    public double run(Properties props) throws FileNotFoundException, IOException, Exception{
        Instances train, test;
        double miF=0;
        for (int i=0; i<trains.size(); i++){
            train = new Instances(trains.get(i));
            test = new Instances(tests.get(i));
            StringToWordVector filter = Tokenisation.WordNgrams(props);
            ConstructionARFF obj = new ConstructionARFF(props);
            train = obj.ConstructionInstances(train);
            test = obj.ConstructionInstances(test);
            filter.setInputFormat(train);
            train = Filter.useFilter(train, filter);
            test = Filter.useFilter(test, filter);
            train.setClass(train.attribute("_class"));
            test.setClass(train.attribute("_class"));
            SMO classifier = new SMO();
            classifier.setC(Double.parseDouble(props.getProperty("SVM.CompexityParameter")));
            classifier.buildClassifier(train);
            Evaluation eTest = new Evaluation(train);
            eTest.evaluateModel(classifier, test);
            miF += eTest.unweightedMicroFmeasure();
        }
        return miF/trains.size();
    }
    
    public PropWithMeasure bestComSVM(Properties props, double measure, PrintWriter Out) throws FileNotFoundException, IOException, Exception{
        Instances train, test;
        ArrayList<Double> alMiF = new ArrayList<Double>();
        double miF;
        for (double c=0; c<=1 ; c=c+0.1){
            miF=0;
            for (int i=0; i<trains.size(); i++){
                train = new Instances(trainsNgrams.get(i));
                test = new Instances(testsNgrams.get(i));
                SMO classifier = new SMO();
                classifier.setC(c);
                classifier.buildClassifier(train);
                Evaluation eTest = new Evaluation(train);
                eTest.evaluateModel(classifier, test);
                miF += eTest.unweightedMicroFmeasure();
            }
            miF=miF/trains.size();
            alMiF.add(miF);
            System.out.println("c="+c+" , result="+miF);
            Out.println("c="+c+" , result="+miF);
            Out.flush();
        }
        double bestRes = Collections.max(alMiF);
        int bestIndex = alMiF.indexOf(bestRes);
        System.out.println(" => Best: "+bestRes+" ("+bestIndex+")");
        Out.println(" => Best: "+bestRes+" ("+bestIndex+")");
        Out.flush();
        try (FileOutputStream out = new FileOutputStream("test/config.properties")) {
            props.setProperty("SVM.CompexityParameter", Double.toString(bestIndex*0.1));
            props.store(out, null);
        }
        
        PropWithMeasure PM = new PropWithMeasure(props, bestRes);
        
        return PM;
    }
    
    public PropWithMeasure bestConfig(Properties prop, double OldMesure, String[] Pretraitements, int SizePretraitements, PrintWriter Out) throws IOException, Exception{
        ArrayList<Double> alMiF = new ArrayList();
        
        int [] TabPretraitements = new int[SizePretraitements] ;
        for (int i=0; i<SizePretraitements; i++)
            TabPretraitements[i] = i ;
        
        ArrayList<ArrayList<Integer>> ArrCombPretraitements = NbreCombinaison(TabPretraitements, SizePretraitements);
        
        double res, bestresult = OldMesure;
        int bestindice = -1 ;
        System.out.println("Nombre de combinaisons possibles : "+ArrCombPretraitements.size());
        for (int i=0; i<ArrCombPretraitements.size(); i++){
            //String source = "test/config.properties" ;
            //String target = "test/config"+i+".properties";
            //CopyFile (source, target);
            Properties propTampon = new Properties(prop);
            //InputStream input = new FileInputStream(target);
            //propTampon.load(input);
            
            FileOutputStream out = new FileOutputStream("tmp.properties");
            ArrayList<Integer> Tmp = ArrCombPretraitements.get(i); //Creation de la liste de combinaison
            for(int j=0; j<Tmp.size(); j++){    //Modification du fichier du configuration
                propTampon.setProperty(Pretraitements[Tmp.get(j)], "yes");
                //System.out.println(propTampon.getProperty(Pretraitements[Tmp.get(j)]));
            }
            propTampon.store(out, null);
            out.close();
            /*for(int j=0; j<Tmp.size(); j++){    //Modification du fichier du configuration
                System.out.print(Pretraitements[Tmp.get(j)]+":"+propTampon.getProperty(Pretraitements[Tmp.get(j)])+" ");
            }*/
            res = run(propTampon);
            if (res > bestresult){
                bestresult = res ; 
                bestindice = i;
            }else if (bestindice != -1){
                if ((Tmp.size() < ArrCombPretraitements.get(bestindice).size()) && (bestresult == res)){
                    bestresult = res ; 
                    bestindice = i;
                }
            }
            System.out.println(i+" : "+res);
            Out.println(i+" : "+res);
            Out.flush();
            
            alMiF.add(res);
        }
        // Find the best configuration
        //double bestRes = Collections.max(alMiF);
        //int bestIndex = alMiF.indexOf(bestRes);
        
        if (bestindice != -1){
            try (FileOutputStream out = new FileOutputStream("test/config.properties")) {
                ArrayList<Integer> Tmp = ArrCombPretraitements.get(bestindice);
                System.out.println("Best config pretraitement trouvée : "+bestindice);
                for(int j=0; j<Tmp.size(); j++){    //Modification du fichier du configuration
                    prop.setProperty(Pretraitements[Tmp.get(j)], "yes");
                }
                prop.store(out, null);
            }
        }
        Out.println("Best indice config : "+bestindice);
        Out.flush();
        
        PropWithMeasure PM = new PropWithMeasure(prop, bestresult);
        
        return PM; 
    }
    
    
    public void saveFile(Instances dataSet, String file) throws IOException{
        ArffSaver saver = new ArffSaver();
         saver.setInstances(dataSet);
         saver.setFile(new File(file));
         saver.writeBatch();
    }
    
    
    public ArrayList<ArrayList<Integer>> NbreCombinaison (int [] Pretraitement, int  n){
        ArrayList<ArrayList<Integer>> Combinaison = new ArrayList<ArrayList<Integer>>();
        
        for(int i=0; i<n ; i++){
            int size = Combinaison.size();
            ArrayList<Integer> tmp = new ArrayList<Integer>();
            tmp.add(i);
            Combinaison.add(tmp);
            //ArrayList<ArrayList<Integer>> TmpCombinaison = new ArrayList<ArrayList<Integer>>();
            for(int j=0; j<size; j++){
                tmp = new ArrayList<Integer>(Combinaison.get(j));
                tmp.add(i);
                Combinaison.add(tmp);                
            }
        }
        return Combinaison;
    }

    public void CopyFile (String source, String destination){
        try {
            FileInputStream ins = null;
            FileOutputStream outs = null;
           File infile =new File(source);
           File outfile =new File(destination);
           ins = new FileInputStream(infile);
           outs = new FileOutputStream(outfile);
           byte[] buffer = new byte[1024];
           int length;

           while ((length = ins.read(buffer)) > 0) {
              outs.write(buffer, 0, length);
           } 
           ins.close();
           outs.close();
           //System.out.println("File copied successfully!!");
        } catch(IOException ioe) {
        }         
    }    
}

class PropWithMeasure {
  public Properties prop;
  public double measure;

    public PropWithMeasure(Properties prop, double measure) {
        this.prop = prop;
        this.measure = measure;
    }

    public double getMeasure() {
        return measure;
    }

    public Properties getProp() {
        return prop;
    }

    public void setMeasure(double measure) {
        this.measure = measure;
    }

    public void setProp(Properties prop) {
        this.prop = prop;
    }
}
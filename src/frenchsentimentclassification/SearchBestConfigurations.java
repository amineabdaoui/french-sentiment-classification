package frenchsentimentclassification;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Properties;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;

/**
 *
 * @author Amine
 */
public class SearchBestConfigurations {
    
    private ArrayList<Instances> trains;
    private ArrayList<Instances> tests;
    
    public SearchBestConfigurations(ArrayList<Instances> trs, ArrayList<Instances> tss){
        trains = new ArrayList<Instances>(trs);
        tests = new ArrayList<Instances>(tss);
    }
    
    public Properties bestNgrams(Properties prop) throws IOException, Exception{
        ArrayList<Properties> alProp = new ArrayList();
        ArrayList<Double> alMiF = new ArrayList();
        int min=Integer.parseInt(prop.getProperty("Ngrams.min"));
        int max=Integer.parseInt(prop.getProperty("Ngrams.max"));
        double res;
        for (int i=min; i<=max; i++){
            for (int j=i; j<=max; j++){
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
            }
        }
        // Find the best configuration
        double bestRes = Collections.max(alMiF);
        int bestIndex = alMiF.indexOf(bestRes);
        
        FileOutputStream out = new FileOutputStream("test/config.properties");
        prop.setProperty("Ngrams.min", alProp.get(bestIndex).getProperty("Ngrams.min"));
        prop.setProperty("Ngrams.max", alProp.get(bestIndex).getProperty("Ngrams.max"));
        prop.store(out, null);
        out.close();
        System.out.println(alProp.get(bestIndex).getProperty("Ngrams.min")+" "+alProp.get(bestIndex).getProperty("Ngrams.max"));

        return prop;
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
    
    public void saveFile(Instances dataSet, String file) throws IOException{
        ArffSaver saver = new ArffSaver();
         saver.setInstances(dataSet);
         saver.setFile(new File(file));
         saver.writeBatch();
    }
    
}

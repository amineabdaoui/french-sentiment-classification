package frenchsentimentclassification;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
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
        double miU=0, miB=0, miUB=0;
        System.out.println(prop.size());
        // evaluate bigrams
        FileOutputStream outB = new FileOutputStream("test/configB.properties");
        Properties propB = prop;
        propB.setProperty("Ngrams.min", "2");
        propB.setProperty("Ngrams.max", "2");
        propB.store(outB, null);
        outB.close();
        miB = runNgrams(propB);
        System.out.println("miB = "+miB);
        // evaluate unigrams
        FileOutputStream outU = new FileOutputStream("test/configU.properties");
        Properties propU = prop;
        propU.setProperty("Ngrams.min", "1");
        propU.setProperty("Ngrams.max", "1");
        propU.store(outU, null);
        outU.close();
        miU = runNgrams(propU);
        System.out.println("miU = "+miU);
        // evaluate unigrams+bigrams
        FileOutputStream outUB = new FileOutputStream("test/configUB.properties");
        Properties propUB = prop;
        propUB.setProperty("Ngrams.min", "1");
        propUB.setProperty("Ngrams.max", "2");
        propUB.store(outUB, null);
        outUB.close();
        miUB = runNgrams(propUB);
        System.out.println("miUB = "+miUB);
        
        FileOutputStream out;
        if (miU >= miB && miU >= miUB){
            out = new FileOutputStream("test/config.properties");
            prop.setProperty("Ngrams.min", "1");
            prop.setProperty("Ngrams.max", "1");
            prop.store(out, null);
            out.close();
        } else if (miB >= miU && miB >= miUB){
            out = new FileOutputStream("test/config.properties");
            prop.setProperty("Ngrams.min", "2");
            prop.setProperty("Ngrams.max", "2");
            prop.store(out, null);
            out.close();
        }
        
        return prop;
    }
    
    public double runNgrams(Properties props) throws FileNotFoundException, IOException, Exception{
        Instances train, test;
        double miF=0;
        for (int i=0; i<trains.size(); i++){
            train = trains.get(i);
            test = tests.get(i);
            System.out.print(test.numAttributes()+"\t");
            StringToWordVector filter = Tokenisation.WordNgrams(props);
            filter.setInputFormat(train);
            train = Filter.useFilter(train, filter);
            test = Filter.useFilter(test, filter);
            System.out.print(test.numAttributes()+"\t");
            train.setClass(train.attribute("_class"));
            test.setClass(train.attribute("_class"));
            SMO classifier = new SMO();
            classifier.buildClassifier(train);
            Evaluation eTest = new Evaluation(train);
            eTest.evaluateModel(classifier, test);
            miF += eTest.unweightedMicroFmeasure();
            System.out.println(test.size()+" "+eTest.unweightedMicroFmeasure());
            saveFile(test,"test/"+props.getProperty("Ngrams.min")+props.getProperty("Ngrams.max")+"-"+i+".arrff");
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

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package frenchsentimentclassification;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;

/**
 *
 * @author Amine
 */
public class Vote {
    
    public static void main(String[] args) throws FileNotFoundException, IOException, Exception {
        BufferedReader r1 = new BufferedReader(new InputStreamReader(new FileInputStream("Results/equipe-4_tache3_run1.csv")));
        BufferedReader r2 = new BufferedReader(new InputStreamReader(new FileInputStream("Results/equipe-4_tache3_run2.csv")));
        BufferedReader r3 = new BufferedReader(new InputStreamReader(new FileInputStream("Results/equipe-4_tache3_run3.csv")));
        
        PrintWriter Out = new PrintWriter("Results/equipe-4_tache3_runVote.csv");
        
        String line1,line2,line3,index,text,classe;
        
        while ((line1=r1.readLine())!=null){
            line2=r2.readLine();
            line3=r3.readLine();
            index=line1.split("\t")[0];
            text=line1.split("\t")[1];
            classe=line1.split("\t")[2];
            if (line2.split("\t")[2].equals(line3.split("\t")[2]) && !line2.split("\t")[2].equals(classe)) Out.println(index+"\t"+text+"\t"+line3.split("\t")[2]);
            else Out.println(index+"\t"+text+"\t"+classe);
        }
        Out.close();
        
        
    }
    
}

import java.awt.*;
import java.io.*;
import javax.swing.*;

/**
 * Displays main drawing window and embeds the pattern drawing component.
 *
 * @author Isaac Baldwin
 */
public class DrawingWindow extends JFrame implements Runnable {
    private final PatternDrawingComponent patternDrawingComponent; //drawing component to embed in the window
    private final JFrame frame; //main JFrame
    private final JLabel counterLabel; //label displaying number of correctly drawn patters
    private Integer[][] pattern; //stage pattern
    private final JLabel successLabel = new JLabel(""); //label of whether pattern was drawn correctly or not
    private final String ID; //participant ID
    private final Integer maxDrawings; //number of drawings for each stage
    private Integer stage; //stage number
    private final ShowPatternWindow patternShow; //show the stage pattern


    /**
     * Instantiates a new Drawing window.
     *
     * @param participantID the participant id
     * @param currentFrame  the current JFrame
     * @param stageNumber   the stage number
     */
    public DrawingWindow(String participantID, JFrame currentFrame, Integer stageNumber) {
        ID = participantID;
        frame = currentFrame;
        maxDrawings = 50; //50 drawings for each stage
        counterLabel = new JLabel("Drawn 0/" + maxDrawings + "        ");
        stage = stageNumber;
        importPattern(stage + "_pattern.txt");
        patternDrawingComponent = new PatternDrawingComponent(pattern, ID, stage, frame);
        patternShow = new ShowPatternWindow(pattern);
        Thread thread = new Thread(this);
        thread.start();
        showPatternToDraw();
    }

    /**
     * Displays the pattern that the participant needs to draw for the current stage
     */
    private void showPatternToDraw() {
        frame.getContentPane().removeAll();
        frame.repaint();
        frame.setTitle("Pattern to Draw");
        JButton bContinue = new JButton("Continue");
        bContinue.addActionListener(e -> {
            frame.getContentPane().removeAll();
            openDrawingWindow();
        });
        frame.add(patternShow, BorderLayout.CENTER);
        frame.add(bContinue, BorderLayout.SOUTH);
        frame.setResizable(false);
        frame.setVisible(true);
    }

    /**
     * Displays the drawing component in the window
     */
    private void openDrawingWindow() {
        frame.repaint();
        frame.setTitle("Pattern drawing");
        frame.setLayout(new BorderLayout());

        JPanel topPanel = new JPanel(new FlowLayout(FlowLayout.CENTER));
        JPanel botPanel = new JPanel(new FlowLayout(FlowLayout.CENTER));
        botPanel.setBorder(BorderFactory.createLineBorder(Color.BLACK));
        topPanel.setBorder(BorderFactory.createLineBorder(Color.BLACK));

        JButton bPattern = new JButton("Show pattern");
        JButton bPause = new JButton("Pause");

        counterLabel.setFont(new Font("Calibri", Font.BOLD, 30));
        successLabel.setFont(new Font("Calibri", Font.BOLD, 30));
        counterLabel.setBounds(20, 50, 300, 100);

        bPattern.addActionListener(e -> showPatternToDraw());
        bPause.addActionListener(e -> pause());

        botPanel.add(bPause);
        botPanel.add(bPattern);

        topPanel.add(counterLabel);
        topPanel.add(successLabel);

        frame.add(topPanel, BorderLayout.NORTH);
        frame.add(patternDrawingComponent, BorderLayout.CENTER);
        frame.add(botPanel, BorderLayout.SOUTH);

        frame.setResizable(false);
        frame.setVisible(true);

    }

    /**
     * Closes the software
     */
    private void finish() {
        System.exit(0);
    }

    /**
     * Imports pattern from given text file and assigns a 2d integer array of the pattern from the file
     * @param fileName name of file which stores the stage pattern
     */
    private void importPattern(String fileName) {
        Integer[][] filePattern = new Integer[3][3]; // initialise stage pattern as empty
        try {
            File file = new File("src\\" + fileName);  // open file
            BufferedReader br = new BufferedReader(new FileReader(file));
            String line;
            int i = 0, x = 0;
            while ((line = br.readLine()) != null) {
                if (x == 3) {
                    i++;
                    x = 0;
                }
                filePattern[i][x] = Integer.parseInt(line);  // convert to integer and add to the pattern
                x++;
            }
        } catch (FileNotFoundException e) {
            System.out.println("File not found");
        } catch (IOException e) {
            e.printStackTrace();
        }
        pattern = filePattern;
    }

    /**
     * Shows information between each drawing session
     */
    private void information() {
        frame.getContentPane().removeAll();
        frame.setTitle("Information");
        JButton button = new JButton("Start pattern drawing");
        button.addActionListener(e -> {
            frame.getContentPane().removeAll();
            new DrawingWindow(ID, frame, stage);
        });
        JLabel text = new JLabel("<html>Thank you for completing the drawings. <br/> Please now complete the next pattern " + "50" + " times.<html>", SwingConstants.CENTER);
        text.setFont(new Font("Calibri", Font.BOLD, 30));
        frame.add(text, BorderLayout.CENTER);
        frame.add(button, BorderLayout.SOUTH);
        frame.setVisible(true);
    }

    /**
     * Displays the pause screen
     */
    private void pause() {
        frame.getContentPane().removeAll();
        frame.repaint();
        frame.setTitle("Paused");
        frame.setLayout(new BorderLayout());

        JButton button = new JButton("Continue");
        JLabel label = new JLabel("<html>You are paused.<br/>You can take a break for as long as you need.</html>", SwingConstants.CENTER);
        button.addActionListener(e -> {
            frame.getContentPane().removeAll();
            openDrawingWindow();
        });
        label.setFont(new Font("Calibri", Font.BOLD, 30));
        frame.add(label, BorderLayout.CENTER);
        frame.add(button, BorderLayout.SOUTH);
        frame.setResizable(false);
        frame.setVisible(true);
    }

    @Override
    public void run() { //allows constant update of graphics on separate thread
        while (true) {
            // if pattern drawn matches the stage pattern
            if (patternDrawingComponent.isCorrect()) {
                // update drawn counter and indication of correctly drawn pattern
                counterLabel.setText("Drawn: " + patternDrawingComponent.getCounter() + "/" + maxDrawings + "        ");
                successLabel.setText("Correct");
                // if all 50 patterns have been drawn
                if (patternDrawingComponent.getCounter().equals(maxDrawings)) {
                    try {
                        patternDrawingComponent.saveToFile();  // data saved to file
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                    // if all stages are complete
                    if (stage == 4) {
                        finish();
                    }
                    stage++;  // increment stage
                    information();
                    break;
                }
            // if pattern drawn does not match the stage pattern
            } else {
                if (patternDrawingComponent.getCounter() == 0) {
                    successLabel.setText("");  //initially not indication given as no patterns have been drawn
                } else {
                    // update indication that pattern is drawn incorrectly
                    successLabel.setText("Incorrect");
                }
            }
        }
    }

}

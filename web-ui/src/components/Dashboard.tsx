import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { UploadCloud, Layers, Cpu, CheckCircle2, Zap } from 'lucide-react';

export default function Dashboard() {
  const [file, setFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [progress, setProgress] = useState(0);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const uploadDataset = () => {
    setIsUploading(true);
    let current = 0;
    const interval = setInterval(() => {
      current += 10;
      setProgress(current);
      if (current >= 100) {
        clearInterval(interval);
        setTimeout(() => setIsUploading(false), 500);
      }
    }, 200);
  };

  return (
    <div className="flex flex-col gap-8 p-6 md:p-12 max-w-7xl mx-auto w-full">
      {/* Header */}
      <header className="flex items-center justify-between border-b pb-6">
        <div>
          <h1 className="text-4xl font-bold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-primary to-accent">
            AutoML Predictor
          </h1>
          <p className="text-muted-foreground mt-2 text-lg">
            Meta-Learning Model Recommender Platform
          </p>
        </div>
        <div className="hidden sm:flex gap-4">
          <Badge variant="outline" className="text-sm px-3 py-1 flex gap-2">
            <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></span>
            Engine Active
          </Badge>
        </div>
      </header>

      {/* Main Grid */}
      <main className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        
        {/* Upload Column */}
        <section className="lg:col-span-1 space-y-6">
          <Card className="border-accent/40 shadow-lg shadow-accent/10">
            <CardHeader className="pb-4">
              <CardTitle className="text-xl flex items-center gap-2">
                <UploadCloud className="text-accent" /> Dataset Selection
              </CardTitle>
              <CardDescription>
                Upload your CSV dataset to receive optimal ML model recommendations.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid w-full max-w-sm items-center gap-1.5">
                <Label htmlFor="dataset">Tabular Dataset</Label>
                <Input id="dataset" type="file" onChange={handleFileChange} accept=".csv" className="cursor-pointer" />
              </div>
              
              {file && (
                <div className="bg-muted p-3 rounded-md text-sm">
                  <p className="font-medium text-foreground truncate">{file.name}</p>
                  <p className="text-muted-foreground">{(file.size / 1024).toFixed(2)} KB</p>
                </div>
              )}

              {isUploading && (
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">Analyzing features...</span>
                    <span>{progress}%</span>
                  </div>
                  <Progress value={progress} className="h-2" />
                </div>
              )}
            </CardContent>
            <CardFooter>
              <Button onClick={uploadDataset} disabled={!file || isUploading} className="w-full">
                {isUploading ? "Processing..." : "Extract Meta-Features & Analyze"}
              </Button>
            </CardFooter>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Extracted Meta-Features</CardTitle>
            </CardHeader>
            <CardContent>
              <ul className="space-y-3">
                <li className="flex items-center justify-between text-sm">
                  <span className="flex items-center gap-2 text-muted-foreground"><Layers size={16}/> Instances</span>
                  <span className="font-semibold">{progress === 100 ? "4,592" : "-"}</span>
                </li>
                <li className="flex items-center justify-between text-sm">
                  <span className="flex items-center gap-2 text-muted-foreground"><Cpu size={16}/> Dimensionality</span>
                  <span className="font-semibold">{progress === 100 ? "0.012" : "-"}</span>
                </li>
                <li className="flex items-center justify-between text-sm">
                  <span className="flex items-center gap-2 text-muted-foreground"><Zap size={16}/> Target Balance</span>
                  <span className="font-semibold">{progress === 100 ? "0.85" : "-"}</span>
                </li>
              </ul>
            </CardContent>
          </Card>
        </section>

        {/* Results Column */}
        <section className="lg:col-span-2 space-y-6">
          <Card className="h-full">
            <CardHeader>
              <CardTitle className="text-2xl flex items-center gap-2">
                <CheckCircle2 className="text-green-500" /> Leaderboard & Recommendations
              </CardTitle>
              <CardDescription>
                Probabilistic ranking of the top base models for your dataset.
              </CardDescription>
            </CardHeader>
            <CardContent>
              {progress < 100 ? (
                <div className="flex flex-col items-center justify-center py-20 text-center text-muted-foreground space-y-4">
                  <div className="w-16 h-16 rounded-full border-4 border-muted border-t-accent animate-spin"></div>
                  <p>Awaiting dataset processing...</p>
                </div>
              ) : (
                <div className="space-y-6">
                  {/* Top Model Showcase */}
                  <div className="bg-gradient-to-br from-card to-accent/5 p-6 rounded-lg border border-accent/20">
                    <div className="flex justify-between items-start mb-4">
                      <div>
                        <Badge className="mb-2 bg-gradient-to-r from-accent to-purple-600 text-white border-0">Top Match</Badge>
                        <h3 className="text-3xl font-bold">XGBoost Classifier</h3>
                      </div>
                      <div className="text-right">
                        <p className="text-sm text-muted-foreground">Confidence</p>
                        <p className="text-2xl font-bold text-green-500">92.4%</p>
                      </div>
                    </div>
                    <p className="text-muted-foreground">
                      Best suited for your dataset due to non-linear feature interactions and high instance count.
                    </p>
                    <div className="mt-4 flex gap-4">
                       <Button variant="default">Deploy Model</Button>
                       <Button variant="outline">View Hyperparameters</Button>
                    </div>
                  </div>

                  <div className="border rounded-md overflow-hidden">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="border-b bg-muted/50 text-left">
                          <th className="p-4 font-medium">Rank</th>
                          <th className="p-4 font-medium">Model</th>
                          <th className="p-4 font-medium">Est. Accuracy</th>
                          <th className="p-4 font-medium">Training Time</th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr className="border-b bg-card hover:bg-muted/30 transition-colors">
                          <td className="p-4 font-semibold">1</td>
                          <td className="p-4 font-medium">XGBoost</td>
                          <td className="p-4 text-green-500 font-medium">92.4%</td>
                          <td className="p-4 text-muted-foreground">Fast (~1.2s)</td>
                        </tr>
                        <tr className="border-b bg-card hover:bg-muted/30 transition-colors">
                          <td className="p-4 font-semibold">2</td>
                          <td className="p-4 font-medium">Random Forest</td>
                          <td className="p-4 text-green-500 font-medium">89.1%</td>
                          <td className="p-4 text-muted-foreground">Medium (~3.5s)</td>
                        </tr>
                        <tr className="bg-card hover:bg-muted/30 transition-colors">
                          <td className="p-4 font-semibold">3</td>
                          <td className="p-4 font-medium">Gradient Boosting</td>
                          <td className="p-4 text-yellow-500 font-medium">85.6%</td>
                          <td className="p-4 text-muted-foreground">Slow (~6.0s)</td>
                        </tr>
                      </tbody>
                    </table>
                  </div>

                </div>
              )}
            </CardContent>
          </Card>
        </section>
        
      </main>
    </div>
  );
}

#!/usr/bin/env perl

# filter VMPeak from statistics:

$filename=$ARGV[0];
while(<STDIN>)
{

    if ($filename =~ /statistics$/)
    {
	if ($_ =~ /^#/)
	{
	    print $_;
	}
	else
	{
	    my @columns = split / /, $_;
	    my $tmp = pop @columns; # remove newline/empty space
	    my $vmpeak = pop @columns; # remove VMPeak
	    print join(' ', @columns), " filtered\n";
	}
    }
    else
    {
	print $_;
    }
}
